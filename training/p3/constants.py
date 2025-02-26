# Number of workers for dataloader
workers = 8

# Batch size during training
# batch_size = 128
# you can set 1024 too
batch_size = 128

# image_size
image_size = 28

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs, we use 150 cuz on test we see that 114 is best
num_epochs = 150

# Learning rate for optimizers
lr = 0.001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
