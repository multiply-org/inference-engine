## Version 0.6

### Improvements and new Features
* Introduced forward models parameter to read registered forward models
* Introduced InferenceWriter to write results from Kafka Inference Engine
* Support Kaska Inference Engine
* Use Aux Data Provider from Multiply Core
* Added Progress logging


## Version 0.5

### New Features
* Deduce destination grid from roi
* Extend state mask with masked out pixels from prior engine
* Do not pass covariance values smaller than 0
* Read saved state mask
* Added profiling

## Version 0.1

### Features
* Introduced Inference Prior to encapsulate Access to Prior Engine or Prior files
* Have Wrapper of Kafka Inference Engine
* Include CLI for accessing kafka wrapping inference engine
* Create state mask if not provided
