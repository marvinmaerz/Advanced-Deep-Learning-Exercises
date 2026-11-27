import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students, s is small offset so noise, because derivate of 0 at t=o, making beta_0 too small
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    # T + 1 boundary points from t = 0 to t = T
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)  # [0.0, 1.0, 2.0, ..., timesteps]

    # noise schduler from equation (17)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    
    # recover per-step betas via beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1}) - under equation (17)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize so alpha_bar_0 = 1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0.0001, 0.999)    # paper after euquation (17) - clip to prevent singularities near t = T where alpha_bar_t -> 0 



def sigmoid_beta_schedule(beta_start, beta_end, timesteps, s_limit=6.0):   # add s_limit to control range of sigmoid function
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x

    t = torch.arange(timesteps, dtype=torch.float32)        # rimesteps t = 0, 1, ..., T-1

    # exercise eq. (10): map t to [-s_limit, +s_limit) where sigmoid transitions smoothly without saturating early
    x = -s_limit + (2 * t / timesteps) * s_limit
    betas = beta_start + torch.sigmoid(x) * (beta_end - beta_start)

    return betas


class Diffusion:

    # TODO (2.4): Adapt all methods in this class for the conditional case. You can use y=None to encode that you want to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps
        self.img_size = img_size
        self.device = device

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps)

        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly use them in the forward pass.
        

        # define alphas and cumulative product
        # TODO
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev= F.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)  # is alpha_t-1 with padding 1.0 at index alpha_0-1

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # TODO
        # Calculations for forward diffusion x_t Eq. 2.2a)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_minus_one = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # TODO
        # paper equation (8)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)                              # 1 / sqrt(alphat_t)
        self.posterior_mean_coef_noise = self.betas / self.sqrt_alphas_cumprod_minus_one    # beta_t / sqrt(1-alpha_t)
        self.sqrt_betas = torch.sqrt(self.betas)                                            # sqrt(beat_t)

        # Added for cosine stability (clamped x_0 prediction -> posterior mean)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        # TODO (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t. Note that x and t both have a batch dimension

        # extract() indexes each 1D schedule tensor by timestep t (shape: (batch_size,))
        # and reshapes the values to (batch_size, 1, 1, 1) so they broadcast across C, H, W of x
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_alphas_cumprod_minus_one_t = extract(
            self.sqrt_alphas_cumprod_minus_one, t, x.shape
        )
        coef1 = extract(self.posterior_mean_coef1, t, x.shape)
        coef2 = extract(self.posterior_mean_coef2, t, x.shape)

        # model(x, t) predicts the noise epsilon_theta(x_t, t) present in the noisy image x;
        # reconstruct x_0 (Ho et al. Eq. 15) and clamp to [-1, 1] to prevent blow-up when beta_t -> 0.999
        predicted_noise = model(x, t)
        x_start = (x - sqrt_alphas_cumprod_minus_one_t * predicted_noise) / sqrt_alphas_cumprod_t
        x_start = x_start.clamp(-1.0, 1.0)

        # Equation 7 / 11 in the paper: estimate the posterior mean of p_theta(x_{t-1} | x_t, x_0)
        model_mean = coef1 * x_start + coef2 * x

        # at t_index = 0 / final step, no additional noise is added
        if t_index == 0:
            return model_mean

        # For t > 0, add scaled Gaussian noise z ~ N(0, I) with std sqrt(posterior_variance_t)
        posterior_std_t = torch.sqrt(extract(self.posterior_variance, t, x.shape))
        noise = torch.randn_like(x)    # randn_like matches shape of x

        # TODO (2.2): The method should return the image at timestep t-1.
        return model_mean + posterior_std_t * noise

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        # TODO (2.2): Implement the full reverse diffusion loop from random noise to an image, iteratively ''reducing'' the noise in the generated image.
        # start from random noise
        img = torch.randn(
            (batch_size, channels, image_size, image_size),     # fill this tupel shape with random numbers
            device=self.device
            )
        imgs = []

        for i in tqdm(                                      # iteratively denoise backwards from timestep T-1 down to 0, displaying a tqdm progress bar
                    reversed(range(0, self.timesteps)),
                    desc="sampling loop time step",
                    total=self.timesteps,
                    leave=False,
                ):
                    t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)  # create a (batch_size,) tensor filled with the current timestep index i
                    img = self.p_sample(model, img, t, i)                                   # perform one reverse diffusion step (x_t -> x_{t-1})
                    imgs.append(img)                                                       

        # TODO (2.2): Return the generated images
        return imgs

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):
        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
             noise = torch.randn_like(x_zero)  # create new noise vector if none is provided

        # implements formula (4) from exercise sheet 
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_zero.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod_minus_one, t, x_zero.shape)

        return sqrt_alphas_cumprod_t * x_zero + sqrt_one_minus_alphas_cumprod_t * noise   # sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * epsilon (noise)

    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1"):
        # TODO (2.2): compute the input to the network using the forward diffusion process and predict the noise using the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero)  # create new noise vector if none is provided

        # create noisy sample x_t using forward diffusion process q(x_t | x_0)
        x_noisy = self.q_sample(x_zero=x_zero, t=t, noise=noise)

        # predict added noise epsilon_theta(x_t, t) using denoising model
        predicted_noise = denoise_model(x_noisy, t)             # predict noise added from t=0 to t

        if loss_type == 'l1':
            # TODO (2.2): implement an L1 loss for this task
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            # TODO (2.2): implement an L2 loss for this task
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss
