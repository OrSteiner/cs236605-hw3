import random


def bin_rand():
    return random.randint(0, 1)


r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    start_seq = ""
    temperature = 0.5
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "My Lord!\n No one can undo your greatness!, This is Trivial.\n"
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Using sequences instead of the whole text:**

**For thr following reasons:**
*   Tokens that are far apart are much less dependent on one another - 
    The farther apart that 2 characters are, the less semantically dependent they are, 
    most of the impact of a character is with descending order within the same word, sentence, or paragraph.
    so we will not want that model will be "punished" (the loss function value to increase)
    for predictions he made a long time ago. 
    (although some of the information still flows in the form of the hidden state)
*   For speeding up training - by only training on sequences and not all the text
    we significantly reduce the time it takes for each batch to be calculated, and for the model to
    change accordingly. Because most of the affect on the prediction is made on the proximity of the 
    sample, we will not loose too much information by training on sequences rather then on the entire dataset.

"""

part1_q2 = r"""
**How does the model remembers more than the sequence length?**

There are 2 forms in witch the model can "remember" this information:
* From the hidden state it receives form the previous batch, that represents the history of the text that was
already processed in the current epoch.
* The model parameters are tuned to the structure of the text, so it knows some of the past of the text 
from the structure of the sequence, without explicitly receiving it.
"""

part1_q3 = r"""
**Not shuffling the batches**

We do not want to shuffle the batches, because we wont to feed them to the model in their original order,
unlike the previous learning tasks that we had, where the order of the input was not important,
here we have meaning in the order of the samples, in the form of the context of the sample, in the text,
and our model will see this context with the hidden state received from the previous sequence.


"""

part1_q4 = r"""
**The T parameter in the hot-softmax sampling**

1) In the training, the temperature will not affect so much the direction of the gradients produced by the loss,
only maybe its magnitude(as the relations between each probability will be similar, and the cross-entropy produced)
but on the sampling we want the text to be more structured, and less random, as we want it to have a similar
structure to a text written by shakespeare, and less like it was randomly generated, and lowering the temperature 
makes it less random. 

2) When the temperature is too high, the text predictions seems to be too random, and noisy,
and it does not look like something intelligent, just random.
The higher the temperatures, the more smooth and uniform the distribution over the vocabulary becomes,
and we are practically sampling at random, without giving extra weight to our model predictions.

3)  When the temperature is too low, the model appears to repeat the same few words over and over again.
this happens because the lower the temperature, the more "rigid" the distribution becomes, and the more deteministic
it becomes (as the option with the highest probability will have a very high probability) and when we get to a certain
sequence of words, the is predicted to be followed with a similar sequence, it easily becomes a loop.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=32,
        h_dim=512, z_dim=32, x_sigma2=0.001,
        learn_rate=2e-3, betas=(0.5, 0.999)
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    # ========================
    return hypers


part2_q1 = r"""
**What does the $\sigma^2$ hyperparameter do?**

The $\sigma^2$ hyperparameter allows us to control the relative influence of the data-loss.
In fact, if we will use a small $sigma^2$ the data-loss effect on the loss value will be higher, and vice-versa.
The more effect the data-loss has on the loss value we allow less randomness to our generator and expect results
that resemble the dataset we have.


"""

part2_q2 = r"""
**Loss term and KL divergence:**

1. The purpose of the reconstruction loss is to increase the probability that the model will reconstruct an image
that resembles the imaged used in the feature extruction (encoding) phase. What we try to achieve is to increase
the model's ability to create the kinds of images that exist in the image dataset.
The purpose of the KL divergence is to measure the distance between the posterior and the prior distributions.
We want to make sure that our encoder's outputs are distributed in a similar way to the prior distribution,
which means the distribution of feature vectors in the latent space.

2. We assume the latent space's distribution is Normal with mean=0 and var=1. By minimizing the KL divergence term we
try to make the posterior distribution as close as possible to the prior. if we will not try to normalize the decoder's
values we will end up at the end of training with informed vectors, that each one represents an image from the dataset.
In fact we try to use normally distributed input vectors to our decoder, which means that we try to use normally
distributed vectors for the training latent space of our model.
By constantly comparing the posterior distribution (the encoder's output) to the normal distribution we make sure
that the features created by the encoder in the features vectors are the one's necessary to create an image that belongs
to our instance space. If our encoder will create too many features, the vectors encoded from the instance space will
not be normally distributed, and therefor we will have high KL divergence value.

3. If we wouldn't use the KL divergence term at all, at the end of the training we will have a decoder that gets a
features vector and returns one of the images in the dataset. In fact, the benefit of using the KL divergence term is to
prevent dataset over-fitting.



"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=32, z_dim=16,
        data_label=1, label_noise=0.2,
        discriminator_optimizer=dict(
            type='SGD',  # Any name in nn.optim like SGD, Adam
            lr=0.003,
            # weight_decay=0.1
        ),
        generator_optimizer=dict(
            type='Adam',  # Any name in nn.optim like SGD, Adam
            lr=0.003,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    # ========================
    return hypers


part3_q1 = r"""
**We will sometimes discard the gradients for the sampling** 

When we want to train the discriminator, and not the generator.
If we do not Do it so, we wil not discard the gradients after the backprop was preformed on
the loss of the discriminator, they will impact the training of the generator, and we don't want that.
We want the generator's gradients to be affected only by it's loss, and not the discriminator's.

"""

part3_q2 = r"""
** 1 - Should we stop when the generator loss becomes very low?**
No, as we see from our training, the low loss does not correlate with good image generated, if is only 
for the optimization algorithm to work well. In supervised learning, small loss meant that we learned well, 
here it might represent a poor discriminator

** 2 - What does it means that the discriminator loss remains constant and the generator loss decreases?**
 
 If the generator loss decreases it means it is fooling the discriminator more. 
 If simultaneously the loss of the discriminator remains constant it has to be that the discriminator
 becomes better at identifying the true images at the same time as the generator becomes better at fooling it.

"""

part3_q3 = r"""

**The differences between the VAE and GAN:**

The images produced by the VAE seems smoother then the images produced
by the GAN.
the GEN generated images seems to have regions with sharp edges, by that I
mean that there is a lot of contrast in certain parts of the image.
But with the VAE, all the transitions seems to be smooth and continues.

This is due to the nature of the loss function.
The loss function on the VAE is the data loss + KL loss, and because that the original picture is smooth, so the loss
that is a smoother function then the loss in the GAN, produces a more consistent gradients, and the resulting model 
produces a smoother image.

ON the other hand, the loss in the GAN is based on the discriminator, that could be a very complex function
(and changes over time), and the gradients could be less consistent like in the VAE loss. 
also because that the discriminator is a convolution decoder, it might focuse on certain features in the feature map
of the image, and that can cause the different "regions" in the image to be generated according to what the 
discriminator focuses on.


"""

# ==============


