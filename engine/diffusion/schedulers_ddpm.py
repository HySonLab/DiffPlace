import torch
import numpy as np


class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        variance_type: str = "fixed_small",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.variance_type = variance_type

        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance[0] = self.posterior_variance[1]

        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        self.timesteps = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps: int, device: str = "cpu"):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = np.arange(0, num_inference_steps) * step_ratio
        timesteps = np.flip(timesteps).copy()
        self.timesteps = torch.from_numpy(timesteps).long().to(device)

        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)

    def _get_variance(self, t: int) -> torch.Tensor:
        if self.variance_type == "fixed_small":
            variance = self.posterior_variance[t]
        elif self.variance_type == "fixed_large":
            variance = self.betas[t]
        else:
            raise ValueError(f"Unknown variance type: {self.variance_type}")
        return variance

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: torch.Generator = None,
        return_dict: bool = False,
    ):
        t = timestep

        pred_original_sample = (
            sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output
        ) / self.sqrt_alphas_cumprod[t]

        if self.clip_sample:
            pred_original_sample = torch.clamp(
                pred_original_sample, -self.clip_sample_range, self.clip_sample_range
            )

        posterior_mean = (
            self.posterior_mean_coef1[t] * pred_original_sample
            + self.posterior_mean_coef2[t] * sample
        )

        posterior_variance = self._get_variance(t)
        posterior_std = torch.sqrt(posterior_variance)

        if t > 0:
            noise = torch.randn(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            prev_sample = posterior_mean + posterior_std * noise
        else:
            prev_sample = posterior_mean

        if return_dict:
            return {
                "prev_sample": prev_sample,
                "pred_original_sample": pred_original_sample,
            }
        return prev_sample, pred_original_sample

    def step_with_guidance(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        guidance_fn=None,
        generator: torch.Generator = None,
    ):
        t = timestep

        pred_original_sample = (
            sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output
        ) / self.sqrt_alphas_cumprod[t]

        if self.clip_sample:
            pred_original_sample = torch.clamp(
                pred_original_sample, -self.clip_sample_range, self.clip_sample_range
            )

        if guidance_fn is not None:
            pred_original_sample = pred_original_sample + guidance_fn(pred_original_sample)

        posterior_mean = (
            self.posterior_mean_coef1[t] * pred_original_sample
            + self.posterior_mean_coef2[t] * sample
        )

        posterior_variance = self._get_variance(t)
        posterior_std = torch.sqrt(posterior_variance)

        if t > 0:
            noise = torch.randn(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            prev_sample = posterior_mean + posterior_std * noise
        else:
            prev_sample = posterior_mean

        return prev_sample, pred_original_sample

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def sample_timesteps(self, batch_size: int, device: str = "cpu") -> torch.Tensor:
        return torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=device, dtype=torch.long
        )

    def get_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        alpha_cumprod = self.alphas_cumprod[timesteps]
        snr = alpha_cumprod / (1 - alpha_cumprod)
        return snr

