use std::time::Instant;

use diffusers::{
    models::{unet_2d::{self, UNet2DConditionModel}, vae::{self, AutoEncoderKL}},
    schedulers::ddim::DDIMScheduler,
    transformers::clip::{self, Tokenizer, ClipTextTransformer},
};
use tch::{nn, nn::Module, Device, Kind, Tensor, vision::image::load};
use anyhow::Result;

enum SDTaskCommand {
    Txt2Img,
    Img2Img,
    Inpainting,
}

// TODO Use the builder pattern to build a task ???
// Even better, build a new type of task little by little, the prompt, that can be processed as soon as it's available, also a mask, and image, ... and finally the rest of the parameters for the steps
struct SDTask<F> where F: Fn(usize, &Tensor) {
    command: SDTaskCommand,
    prompt: String,
    negative_prompt: String,
    image: Option<Tensor>,
    width: usize,
    height:  usize,
    guidance_scale: f64,
    strength: f64,
    n_steps: usize,
    seed: i64,
    callback: F
}

struct SDPipelineConfig {
    device: Device,
    bpe_path: String,
    unet_weights_path: String,
    clip_weights_path: String,
    vae_weights_path: String,
    sliced_attention_size: Option<i64>,
    inpainting_model: bool,
    kind: Kind,
}

struct SDPipeline {
    device: Device,
    tokenizer: Tokenizer,
    text_model:  ClipTextTransformer,
    vae: AutoEncoderKL,
    unet: UNet2DConditionModel,
    inpainting_model: bool,
    kind: Kind,
}

impl SDPipeline {

    pub fn new(config: SDPipelineConfig) -> Result<Self> {
        let device = config.device;
        let kind = config.kind;
        let inpainting_model = config.inpainting_model;
        let tokenizer = clip::Tokenizer::create(config.bpe_path)?;
        let text_model = SDPipeline::build_clip_transformer(&config.clip_weights_path, device, kind)?;
        let vae = SDPipeline::build_vae(&config.vae_weights_path, device, kind)?;
        let unet = SDPipeline::build_unet(&config.unet_weights_path, device, config.sliced_attention_size, inpainting_model, kind)?;
        Ok(SDPipeline { device, tokenizer, text_model, vae, unet, inpainting_model, kind })
    }

    fn build_clip_transformer(clip_weights: &str, device: Device, kind: Kind) -> Result<clip::ClipTextTransformer> {
        let mut vs = nn::VarStore::new(device);
        let text_model = clip::ClipTextTransformer::new(vs.root());
        vs.load(clip_weights)?;
        vs.set_kind(kind);
        Ok(text_model)
    }
    
    fn build_vae(vae_weights: &str, device: Device, kind: Kind) -> Result<vae::AutoEncoderKL> {
        let mut vs_ae = nn::VarStore::new(device);
        let autoencoder_cfg = vae::AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
        };
        let autoencoder = vae::AutoEncoderKL::new(vs_ae.root(), 3, 3, autoencoder_cfg);
        vs_ae.load(vae_weights)?;
        vs_ae.set_kind(kind);
        Ok(autoencoder)
    }
    
    fn build_unet(unet_weights: &str, device: Device, sliced_attention_size: Option<i64>, inpainting_model: bool, kind: Kind) -> anyhow::Result<unet_2d::UNet2DConditionModel> {
        let mut vs_unet = nn::VarStore::new(device);
        let unet_cfg = unet_2d::UNet2DConditionModelConfig {
            attention_head_dim: 8,
            blocks: vec![
                unet_2d::BlockConfig { out_channels: 320, use_cross_attn: true },
                unet_2d::BlockConfig { out_channels: 640, use_cross_attn: true },
                unet_2d::BlockConfig { out_channels: 1280, use_cross_attn: true },
                unet_2d::BlockConfig { out_channels: 1280, use_cross_attn: false },
            ],
            center_input_sample: false,
            cross_attention_dim: 768,
            downsample_padding: 1,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            layers_per_block: 2,
            mid_block_scale_factor: 1.,
            norm_eps: 1e-5,
            norm_num_groups: 32,
            sliced_attention_size,
        };
        let in_channels = if inpainting_model { 9 } else { 4 };
        let unet = unet_2d::UNet2DConditionModel::new(vs_unet.root(), in_channels, 4, unet_cfg);
        vs_unet.load(unet_weights)?;
        vs_unet.set_kind(kind);
        Ok(unet)
    }    

    pub fn run<F>(&self, task: SDTask<F>) -> Result<Tensor> 
        where F: Fn(usize, &Tensor)
    {
        // TODO Split the different parts of running into different stages to process as soon as possible and reduce latency on an editor

        let _no_grad_guard = tch::no_grad_guard();

        let text_embeddings = self.generate_embeddings(&task.prompt, &task.negative_prompt)?;

        let scheduler = DDIMScheduler::new(task.n_steps, 1000, Default::default());

        let mut timesteps = scheduler.timesteps();

        let mut latents = match task.command {
            SDTaskCommand::Txt2Img => {
                tch::manual_seed(task.seed);
                Tensor::randn(&[1, 4, (task.height / 8) as i64, (task.width / 8) as i64], (self.kind, self.device))
            },
            SDTaskCommand::Img2Img => {
                let image = task.image.unwrap();
                let init_latents = self.vae_encode_and_sample(&image);

                // Get the original timestep using init_timestep
                let offset = 0usize; // TODO Steps offset from the scheduler, assume 0?
                let init_timestep_idx = (task.n_steps as f64 * task.strength) as usize + offset;
                let init_timestep_idx = init_timestep_idx.min(task.n_steps);

                let init_timestep_tensor = Tensor::of_slice(&[timesteps[timesteps.len() - init_timestep_idx] as i64]);

                // Add noise to latents using the timesteps (init_timestep_tensor)
                tch::manual_seed(task.seed); // We need the seed for the noise, not for the sample of the provided image
                let noise = Tensor::randn(&init_latents.size(), (self.kind, self.device));
                let latents = scheduler.add_noise(&init_latents, &noise, &init_timestep_tensor);

                // Update timesteps
                let t_start = (task.n_steps - init_timestep_idx + offset).max(0);
                timesteps = &scheduler.timesteps()[t_start..];

                latents
            },
            SDTaskCommand::Inpainting => {
                // TODO Like Img2Img but without adding the noise ????
                // Also prepare mask (encode with vae, ...) and masked image
                // https://github.com/huggingface/diffusers/blob/2a0c823527694058d410ed6f91b52e7dd9f94ebe/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py
                unimplemented!()
            },
        };

        let mut mask = Tensor::new();
        let mut masked_image_latents = Tensor::new();

        if self.inpainting_model {
            match task.command {
                SDTaskCommand::Txt2Img | SDTaskCommand::Img2Img => {
                    // TODO This could be reused as long as the dimensions are the same

                    // This way of using the inpainting model in Txt2Img and Img2Img seems to work well.
                    // But in the Img2Img case seems to differ from Automatic1111 results, but they don't seem better.
                    // On Txt2Img they see identical in the examples tested

                    // The mask is all white because we want to repaint everything
                    mask = Tensor::ones(&[2, 1, (task.height / 8) as i64, (task.width / 8) as i64], (self.kind, self.device));
                    // The masked image is all zeros because we don't keep anything outside the mask
                    let masked_image = Tensor::zeros(&[2, 3, task.height as i64, task.width as i64], (self.kind, self.device));
                    // Encode the masked image
                    masked_image_latents = self.vae_encode_and_sample(&masked_image);
                },
                SDTaskCommand::Inpainting => {
                    // TODO Prepare the mask and masked_image_latents using the provided mask
                    unimplemented!()
                },
            };
        }
    
        for (timestep_index, &timestep) in timesteps.iter().enumerate() {
            // Expand the latents because we are doing classifier free guidance
            let mut latent_model_input = Tensor::cat(&[&latents, &latents], 0);

            // If we are using an inpainting model, concatenate the mask and masked_image_latents
            if self.inpainting_model {
                latent_model_input = Tensor::cat(&[&latent_model_input, &mask, &masked_image_latents], 1);
            }

            // Predict the noise residual
            let noise_pred = self.unet.forward(&latent_model_input, timestep as f64, &text_embeddings);

            // Perform guidance
            let noise_pred = noise_pred.chunk(2, 0);
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            let noise_pred = noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * task.guidance_scale;

            // TODO Do inpainting without the inpainting model

            // Compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(&noise_pred, timestep, &latents);

            // Call the callback
            (task.callback)(timestep_index, &latents);
        }

        Ok(self.generate_image(latents))
    }
    
    fn generate_tokens(&self, prompt: &str) -> Result<Tensor> {
        let tokens = self.tokenizer.encode(&prompt)?;
        let tokens: Vec<i64> = tokens.into_iter().map(|x| x as i64).collect();
        Ok(Tensor::of_slice(&tokens).view((1, -1)).to(self.device))
    }

    
    fn generate_embeddings(&self, prompt: &str, negative_prompt: &str) -> Result<Tensor> {

        let tokens = self.generate_tokens(prompt)?;
        let uncond_tokens = self.generate_tokens(negative_prompt)?;

        let text_embeddings = self.text_model.forward(&tokens);
        let uncond_embeddings = self.text_model.forward(&uncond_tokens);
        Ok(Tensor::cat(&[uncond_embeddings, text_embeddings], 0).to(self.device)) // Classifier-free guidance
    }

    fn generate_image(&self, latents: Tensor) -> Tensor {
        let image = self.vae.decode(&(&latents / 0.18215));
        let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
        (image * 255.).to_kind(Kind::Uint8)
    }

    fn vae_encode_and_sample(&self, image: &Tensor) -> Tensor {
            let latents_dist = self.vae.encode(&image);
            // Sample the DiagonalGaussianDistribution - https://github.com/huggingface/diffusers/blob/2a0c823527694058d410ed6f91b52e7dd9f94ebe/src/diffusers/models/vae.py#L357
            let latents_dist_chunk =  latents_dist.chunk(2, 1);
            let mean = &latents_dist_chunk[0];
            let logvar = latents_dist_chunk[1].clamp(-30.0, 20.0);
            let std = (logvar * 0.5).exp();
            let sample = Tensor::randn(&mean.size(), (self.kind, self.device));
            (mean + std * sample) * 0.18215
    }
}


fn main() -> anyhow::Result<()> {
    tch::maybe_init_cuda();

    let device = Device::cuda_if_available();
    let kind = Kind::Half;

    let config = SDPipelineConfig {
        device,
        bpe_path: "data/bpe_simple_vocab_16e6.txt".to_owned(),
        unet_weights_path: "data/unet_inpainting_v1_5_fp16.ot".to_owned(),
        clip_weights_path: "data/clip_v1_5_fp16.ot".to_owned(),
        vae_weights_path: "data/vae_v1_5_fp16.ot".to_owned(),
        sliced_attention_size: Some(1),
        inpainting_model: true, // When using the inpainting unet, you can still use the base clip and vae
        kind,
    };

    let sd = SDPipeline::new(config)?;

    loop { // Generate images until killed
        println!("Starting to generate the image ...");

        let start = Instant::now();

        let image = load("input_image.png")?.to_kind(kind).to_device(device);
        let image = (image.unsqueeze(0) / 255.) * 2.0 - 1.0;

        let task = SDTask {
            command: SDTaskCommand::Txt2Img,
            prompt: "A very realistic photo of a rusty robot walking on a sandy beach".to_owned(),
            negative_prompt: "".to_owned(),
            image: Some(image),
            width: 512,  height: 512,
            guidance_scale: 7.5, n_steps: 30, strength: 1.0,
            seed: 321,
            callback: |i, _| {
                if i > 0 && i % 10 == 0 {
                    let elapsed = start.elapsed();
                    println!("Step {} - Avg speed {} it/s", i, i as f32 / elapsed.as_millis() as f32 * 1000.0);
                }
            }
        };
    
        let image = sd.run(task)?;
    
        println!("Total generation time {}s", start.elapsed().as_millis() as f32 / 1000.0);
    
        tch::vision::image::save(&image, "image.png")?;

        drop(image);

        break;
    }

    Ok(())
}
