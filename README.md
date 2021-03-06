# DaisyGANv1

This is a [Generative adversarial network](https://en.wikipedia.org/wiki/Generative_adversarial_network) which digests character streams.

The idea here was to take a stream of messages from a specific user. In this case, a user on Telegram named Daisy was chosen due to her unique writing style and quote-like sentences; Daisy is prone to re-using the same unique typo's and miss-spellings in her messages, making her messages easier to identify. Thus it was a fair argument to suggest her messages would make a good test-case for a neural network to discriminate if selected messages were from her - or not.

With most messages being under 256 characters I decided that this was a good cache-friendly input length. These characters are fed into 256 fully connected perceptrons, which feed into a fully connected hidden layer of 8192 perceptrons, which then finally feeds back into a fully connected output layer of 256 perceptrons. I found the arctan activation function was most optimal in this network; as such, every neuron uses the arctan activator. I did experiment with other activators and using arctan only on the output layer.

The idea you can see is simple, take in a character stream, output a stream.

Generally, this model did not perform well; and the generator failed to produce any intelligible output while the discriminator only just produced an adequate classification.

Performance could be increased by only allowing lowercase alphanumeric input characters of 0x61 to 0x7A and 0x30 to 0x39; however, I propose the next attempt would yield better results disseminating messages in chunks of lower case words with a post-process to re-punctuate.
 
## Example Usage
```
./cfdgan retrain
./cfdgan "this is an example scentence"
./cfdgan ask
```
