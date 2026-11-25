import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    pass


def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x
    pass


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

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        # TODO (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t. Note that x and t both have a batch dimension

        # extract() indexes each 1D schedule tensor by timestep t (shape: (batch_size,))
        betas_t = extract(self.betas, t, x.shape)
        # reshapes the values to (batch_size, 1, 1, 1) so they broadcast across C, H, W of x
        sqrt_alphas_cumprod_minus_one_t = extract(
            self.sqrt_alphas_cumprod_minus_one, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper, estimate the mean of p_theta(x_{t-1} | x_t)
        # model(x, t) predicts the noise epsilon_theta(x_t, t) present in the noisy image x;
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_alphas_cumprod_minus_one_t  # model(x, t) is the predicted noise
        )

        # at t_index = 0 / final step, no additional noise is added
        if t_index == 0:
            return model_mean

        # For t > 0, add scaled Gaussian noise z ~ N(0, I) with std sqrt(beta_t)
        posterior_variance_t = extract(self.sqrt_betas, t, x.shape)
        noise = torch.randn_like(x)    # randn_like matches shape of x
        
        # TODO (2.2): The method should return the image at timestep t-1.
        return model_mean + posterior_variance_t * noise     # in exercise equation (8) sqrt(beta_z) * z

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
        pass

    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1"):
        # TODO (2.2): compute the input to the network using the forward diffusion process and predict the noise using the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.

        if loss_type == 'l1':
            # TODO (2.2): implement an L1 loss for this task
            loss = None
        elif loss_type == 'l2':
            # TODO (2.2): implement an L2 loss for this task
            loss = None
        else:
            raise NotImplementedError()

        return loss
