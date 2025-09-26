import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
from utils import general
from networks import networks, cheb_kan
from objectives import ncc
from objectives import regularizers


TERMINATION_AT = 0.75
class ImplicitRegistrator3d:
    """This is a class for registrating implicitly represented images."""

    def __call__(
        self, coordinate_tensor=None, output_shape=(28, 28), dimension=0, slice_pos=0
    ):
        """Return the image-values for the given input-coordinates."""

        # Use standard coordinate tensor if none is given
        if coordinate_tensor is None:
            coordinate_tensor = self.make_coordinate_slice(
                output_shape, dimension, slice_pos
            )

        output = self.network(coordinate_tensor)

        # Shift coordinates by 1/n * v
        coord_temp = torch.add(output, coordinate_tensor)

        transformed_image = self.transform_no_add(coord_temp)
        return (
            transformed_image.cpu()
            .detach()
            .numpy()
            .reshape(output_shape[0], output_shape[1])
        )

    def __init__(self, moving_image, fixed_image, **kwargs):
        """Initialize the learning model."""

        self.set_default_arguments()

        self.args.update(kwargs)
        for key in self.args.keys():
            setattr(self, key, self.args[key])

        torch.manual_seed(self.seed)

        self.save_folder += "/"

        # Make loss list to save losses
        self.loss_list = [0 for _ in range(self.epochs)]
        self.data_loss_list = [0 for _ in range(self.epochs)]
        if self.network_from_file is None:
            if self.network_type == "kan":
                self.network = cheb_kan.ChebyKAN(layers_hidden=self.kan_layers, degree=self.degree)

            elif self.network_type == "a_kan":
                self.network = cheb_kan.AChebyKAN(
                    layers_hidden=self.kan_layers, degree=self.degree, init_cfg=self.init_cfg)

            elif self.network_type == "rand_kan":
                self.network = cheb_kan.RandChebyKAN(
                    layers_hidden=self.kan_layers, degree=self.degree, init_cfg=self.init_cfg)
                
            elif self.network_type == "MLP":
                self.network = networks.MLP(self.layers)

            else:
                self.network = networks.Siren(self.layers, self.weight_init, self.omega)

            if self.verbose:
                print(
                    "Network contains {} trainable parameters.".format(
                        general.count_parameters(self.network)
                    )
                )
        else:
            self.network = torch.load(self.network_from_file)
            if self.gpu:
                self.network.cuda()

        # Choose the optimizer
        if self.optimizer_arg.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )

        elif self.optimizer_arg.lower() == "adam":
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        else:
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )
            print(
                "WARNING: "
                + str(self.optimizer_arg)
                + " not recognized as optimizer, picked SGD instead"
            )

        if self.scheduler_type == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                                 max_lr=1e-4, 
                                                                 div_factor=1, 
                                                                 final_div_factor=10, 
                                                                 pct_start=0.5, 
                                                                 total_steps=self.epochs)
        elif self.scheduler_type == 'exp':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
                                                               lambda x: 0.025**min(x/self.epochs, 1))
        else:
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, 
                                                                 factor=1, \
                                                                 total_iters=1)

        # Choose the loss function
        if self.loss_function.lower() == "mse":
            self.criterion = nn.MSELoss()

        elif self.loss_function.lower() == "ncc":
            self.criterion = ncc.NCC()

        else:
            self.criterion = nn.MSELoss()
            print(
                "WARNING: "
                + str(self.loss_function_arg)
                + " not recognized as loss function, picked MSE instead"
            )

        # Move variables to GPU
        if self.gpu:
            self.network.cuda()

        # Initialization
        self.moving_image = moving_image
        self.fixed_image = fixed_image
        self.possible_coordinate_tensor = general.make_masked_coordinate_tensor_3d(
            self.mask, self.fixed_image.shape
        )

        if self.gpu:
            self.moving_image = self.moving_image.cuda()
            self.fixed_image = self.fixed_image.cuda()

    def cuda(self):
        """Move the model to the GPU."""
        self.network.cuda()
        self.moving_image.cuda()
        self.fixed_image.cuda()

    def set_default_arguments(self):
        """Set default arguments."""

        # Inherit default arguments from standard learning model
        self.args = {}

        # Define the value of arguments
        self.args["mask"] = None
        self.args["mask_2"] = None

        self.args["method"] = 1

        self.args["lr"] = 0.00001
        self.args["batch_size"] = 10000
        self.args["layers"] = [3, 256, 256, 256, 3]
        self.args["kan_layers"] = [3, 256, 256, 256, 3]
        self.args["velocity_steps"] = 1

        # Define argument defaults specific to this class
        self.args["output_regularization"] = False
        self.args["alpha_output"] = 0.2
        self.args["reg_norm_output"] = 1

        self.args["compute_tv_jdet_loss"] = False
        self.args["alpha_jacobian"] = 5.0
        self.args["alpha_tv"] = 0.4

        self.args["hyper_regularization"] = False
        self.args["alpha_hyper"] = 0.25

        self.args["bending_regularization"] = False
        self.args["alpha_bending"] = 10.0

        self.args["diffusion_regularization"] = False
        self.args["alpha_diffusion"] = 1.0

        self.args["image_shape"] = (200, 200)

        self.args["network_from_file"] = None

        self.args["epochs"] = 2500
        self.args["log_interval"] = self.args["epochs"] // 4
        self.args["verbose"] = True
        self.args["save_folder"] = "output"

        self.args["network_type"] = "MLP"

        self.args["gpu"] = torch.cuda.is_available()
        self.args["optimizer_arg"] = "Adam"
        self.args["loss_function"] = "ncc"
        self.args["momentum"] = 0.5

        self.args["positional_encoding"] = False
        self.args["weight_init"] = True
        self.args["omega"] = 32

        self.args["seed"] = 1
        self.args['scheduler_type'] = None

    def get_temperature(self, epoch):
        if epoch < int(TERMINATION_AT * self.epochs):
            return (1 - epoch / (TERMINATION_AT * self.epochs)) ** 0.9
        else:
            return 0.0
        
    def training_iteration(self, epoch):
        """Perform one iteration of training."""

        # Reset the gradient
        self.network.train()

        loss = 0
        indices = torch.randperm(
            self.possible_coordinate_tensor.shape[0], device="cuda"
        )[: self.batch_size]
        coordinate_tensor = self.possible_coordinate_tensor[indices, :]
        coordinate_tensor = coordinate_tensor.requires_grad_(True)
        if self.learn_logits and epoch > int(TERMINATION_AT * self.epochs) :
            for param in self.network.deep_prior.parameters():
                param.requires_grad = False
            self.learn_logits = False

        output = self.network(coordinate_tensor, self.get_temperature(epoch))
        coord_temp = torch.add(output, coordinate_tensor)
        output = coord_temp
        
        self.transformed_image = self.transform_no_add(coord_temp)
        fixed_image = general.fast_trilinear_interpolation_3d(
            self.fixed_image,   
            coordinate_tensor[:, 0],
            coordinate_tensor[:, 1],
            coordinate_tensor[:, 2],
        )

        loss += self.criterion(self.transformed_image, fixed_image)
        if self.verbose:
            self.data_loss_list[epoch] = loss.detach().cpu().numpy()

        output_rel = torch.subtract(output, coordinate_tensor)

        if self.tv_jac_regularization:
            loss += regularizers.compute_tv_jdet_loss(
                coordinate_tensor, 
                output_rel, 
                jac_weight=self.alpha_jacobian, 
                tv_weight=self.alpha_tv, 
                eps=self.loss_eps
            )
        if self.hyper_regularization:
            loss += self.alpha_hyper * regularizers.compute_hyper_elastic_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
        if self.bending_regularization:
            loss += self.alpha_bending * regularizers.compute_bending_energy_3d(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )

        if self.diffusion_regularization:
            loss += self.alpha_diffusion * regularizers.compute_diffusion_loss_3d(
                coordinate_tensor, output_rel
            )

        # Perform the backpropagation and update the parameters accordingly

        for param in self.network.parameters():
            param.grad = None
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # Store the value of the total loss
        if self.verbose:
            self.loss_list[epoch] = loss.detach().cpu().numpy()

    def transform(
        self, transformation, coordinate_tensor=None, moving_image=None, reshape=False
    ):
        """Transform moving image given a transformation."""

        # If no specific coordinate tensor is given use the standard one of 28x28
        if coordinate_tensor is None:
            coordinate_tensor = self.coordinate_tensor

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image

        # From relative to absolute
        transformation = torch.add(transformation, coordinate_tensor)
        return general.fast_trilinear_interpolation_3d(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
            transformation[:, 2],
        )

    def transform_no_add(self, transformation, moving_image=None, reshape=False):
        """Transform moving image given a transformation."""

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image

        return general.fast_trilinear_interpolation_3d(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
            transformation[:, 2],
        )

    def fit(self, epochs=None):
        """Train the network."""

        if epochs is None:
            epochs = self.epochs

        torch.manual_seed(self.args["seed"])

        if not len(self.loss_list) == epochs:
            self.loss_list = [0 for _ in range(epochs)]
            self.data_loss_list = [0 for _ in range(epochs)]

        if self.verbose:
            tgen = tqdm.tqdm(range(epochs), ncols=100)
        else:
            tgen = range(epochs)
        if self.network_type == 'a_chebkan':
            self.learn_logits = True
        else:
            self.learn_logits = False
        for i in tgen:
            self.training_iteration(i)

        if self.verbose:
            plt.plot(self.loss_list)
            plt.savefig(f'loss_{self.network_type}.png')
            plt.close()