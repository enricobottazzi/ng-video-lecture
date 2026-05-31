
# nanogpt-lecture

Code created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT. Publishing here as a Github repo so people can easily hack it, walk through the `git log` history of it, etc.

NOTE: sadly I did not go too much into model initialization in the video lecture, but it is quite important for good performance. The current code will train and work fine, but its convergence is slower because it starts off in a not great spot in the weight space. Please see [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) for `# init all weights` comment, and especially how it calls the `_init_weights` function. Even more sadly, the code in this repo is a bit different in how it names and stores the various modules, so it's not possible to directly copy paste this code here. My current plan is to publish a supplementary video lecture and cover these parts, then I will also push the exact code changes to this repo. For now I'm keeping it as is so it is almost exactly what we actually covered in the video.

### License

MIT

### Optimization

Bench: inference with 8 tokens prefill, 24 tokens generated

Initial bench: 0.987s (506.6 tok/s)

The initial `generate` function, corresponding to this [repo](https://github.com/karpathy/ng-video-lecture), is ineffiicent for two reasons:
- At each call, re-computes the key and value matrices for all the tokens in the context. A more efficient approach would be to cache these matrices and reuse them for each call.
- At each call, fetch the logits for all the tokens in the context. A more efficient approach would be to fetch the logits for only the last token and discard the rest.

New bench: inference: 0.021s (1165.0 tok/s)