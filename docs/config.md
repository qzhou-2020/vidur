# understand the configuration options for Vidur

## Cluster

- `cluster_config_num_replicas`: number of replicas in the cluster.

## replica

hardware:

- `replica_config_device`: device type of the replica.
    - `a40`: A40 GPU: 40GB
    - `a100`: A100 GPU: 80GB
    - `h100`: H100 GPU: 80GB
- `replica_config_model_name`: name of the model to use.
    - `codellama/CodeLlama-34b-instruct-hf`: CodeLlama-34B
    - `meta-llama/Llama-2-7b-hf`: LLaMA2-7B
    - `meta-llama/Llama-2-70b-hf`: LLaMA2-70B
    - `meta-llama/Meta-Llama-3-8b-hf`: LLaMA3-8B
    - `meta-llama/Meta-Llama-3-70b-hf`: LLaMA3-70B
    - `internlm/Internlm-20b-hf`: InternLM-20B
    - `microsoft/phi-2`: Phi-2
    - `Qwen/Qwen-72B`: Qwen-72B
- `replica_config_network_device`: network device to use for the replica.
    - `a40_pairwise_nvlink`: two linked A40 GPUs
    - `a100_pairwise_nvlink`: two linked A100 GPUs
    - `a100_dgx`: single A100 GPU
    - `h100_pairwise_nvlink`: two linked H100 GPUs
    - `h100_dgx`: single H100 GPU
- `replica_config_tensor_parallel_size`: default 1. tensor parallel size of the replica.
- `replica_config_num_pipeline_stages`: default 1. number of pipeline stages of the replica.
- `replica_config_memory_margin_fraction`: default 0.1. fraction of memory to reserve for other overheads (e.g. gradients, activations, etc.)

firmware:

- `global_scheduler_config_type`: type of global scheduler to use.
    - `round_robin`: round-robin scheduler
    - `random`: random scheduler
    - `lor`: LOR scheduler

- `replica_scheduler_config_type`: type of replica scheduler to use.
    - `vllm`: vLLM scheduler
    - `lightllm`: LightLLM scheduler
    - `orca`: ORCA scheduler
    - `sarathi`: Sarathi scheduler
    - `faster_transformer`: FasterTransformer scheduler

Global scheduler distributes requests to replicas. Replica scheduler monitors the replica's GPU usage and decides when to send a new batch to the GPU. For each replica type, there will be more configurations.

### vLLM scheduler

- `vllm_scheduler_config_block_size`: block size to use for the vLLM scheduler.
- `vllm_scheduler_config_watermark_blocks_fraction`: fraction of the buffer to use as the watermark for the vLLM scheduler.
- `vllm_scheduler_config_num_blocks`: number of blocks to use for the vLLM scheduler.
- `vllm_scheduler_config_max_tokens_in_batch`: maximum number of tokens in a batch for the vLLM scheduler.

### lightllm scheduler

- `lightllm_scheduler_config_batch_size_cap`: maximum batch size to use for the lightllm scheduler.
- `lightllm_scheduler_config_block_size`: block size to use for the lightllm scheduler.
- `lightllm_scheduler_config_watermark_blocks_fraction`: fraction of the buffer to use as the watermark for the lightllm scheduler.
- `lightllm_scheduler_config_num_blocks`: number of blocks to use for the lightllm scheduler.
- `lightllm_scheduler_config_max_tokens_in_batch`: maximum number of tokens in a batch for the lightllm scheduler.
- `lightllm_scheduler_config_max_waiting_iters`: maximum number of iterations to wait for the lightllm scheduler.

### ORCA scheduler

- `orca_scheduler_config_batch_size_cap`: maximum batch size to use for the ORCA scheduler.
- `orca_scheduler_config_watermark_blocks_fraction`: fraction of the buffer to use as the watermark for the ORCA scheduler.
- `orca_scheduler_config_num_blocks`: number of blocks to use for the ORCA scheduler.

### Sarathi scheduler

- `sarathi_scheduler_config_batch_size_cap`: maximum batch size to use for the Sarathi scheduler.
- `sarathi_scheduler_config_block_size`: block size to use for the Sarathi scheduler.
- `sarathi_scheduler_config_watermark_blocks_fraction`: fraction of the buffer to use as the watermark for the Sarathi scheduler.
- `sarathi_scheduler_config_num_blocks`: number of blocks to use for the Sarathi scheduler.
- `sarathi_scheduler_config_chunk_size`: chunk size to use for the Sarathi scheduler.

### faster_transformer scheduler

- `faster_transformer_scheduler_config_batch_size_cap`: maximum batch size to use for the faster_transformer scheduler.
- `faster_transformer_scheduler_config_block_size`: block size to use for the faster_transformer scheduler.
- `faster_transformer_scheduler_config_watermark_blocks_fraction`: fraction of the buffer to use as the watermark for the faster_transformer scheduler.
- `faster_transformer_scheduler_config_num_blocks`: number of blocks to use for the faster_transformer scheduler.


## request generator

request generator generates requests to the cluster. One need to choose from a `synthetic` or `trace-replay` typed request generator.

- `request_generator_config_type`: type of request generator to use.
    - `synthetic`: synthetic request generator
    - `trace-replay`: trace replay request generator

### synthetic request generator

- `synthetic_request_generator_config_seed`: seed for the random number generator.

The following two parameters are used to control the how many requests to generate. Duration has priority over number of requests.

- `synthetic_request_generator_config_num_requests`: number of requests to generate.
- `synthetic_request_generator_config_duration`: duration of the request generation.

Synthetic request generator generates requests with random sizes and random inter-request intervals. These are controlled by `length_generator` and `interval_generator` respectively.

#### length generator

- `length_generator_config_type`: type of length generator to use.
    - `trace`: trace-based length generator
    - `uniform`: uniform length generator
    - `zipf`: zipfian length generator
    - `fixed`: fixed length generator

for trace-based length generator,

- `trace_request_length_generator_config_seed`: seed for the random number generator.
- `trace_request_length_generator_config_trace_file`: path to the trace file.
    - `./data/processed_traces`
- `trace_request_length_generator_config_max_tokens`: maximum number of tokens in a request.
- `trace_request_length_generator_config_prefill_scale_factor`: scale factor for the prefill requests.
- `trace_request_length_generator_config_decode_scale_factor`: scale factor for the decode requests.
- `trace_request_length_generator_config_wrap_around`: whether to wrap around the trace file.

for uniform length generator,

- `uniform_request_length_generator_config_seed`: seed for the random number generator.
- `uniform_request_length_generator_config_max_tokens`: maximum number of tokens in a request.
- `uniform_request_length_generator_config_min_tokens`: minimum number of tokens in a request.
- `uniform_request_length_generator_config_prefill_to_decode_ratio`: ratio of prefill to decode tokens.

for zipfian length generator,

- `zipf_request_length_generator_config_seed`: seed for the random number generator.
- `zipf_request_length_generator_config_max_tokens`: maximum number of tokens in a request.
- `zipf_request_length_generator_config_theta`: theta for the zipf distribution.
- `zipf_request_length_generator_config_scramble`: whether to scramble the requests.
- `zipf_request_length_generator_config_min_tokens`: minimum number of tokens in a request.
- `zipf_request_length_generator_config_prefill_to_decode_ratio`: ratio of prefill to decode tokens.

for fixed length generator,

- `fixed_request_length_generator_config_seed`: seed for the random number generator.
- `fixed_request_length_generator_config_max_tokens`: maximum number of tokens in a request.
- `fixed_request_length_generator_config_prefill_tokens`: number of prefill tokens in a request.
- `fixed_request_length_generator_config_decode_tokens`: number of decode tokens in a request.

#### interval generator

- `interval_generator_config_type`: type of interval generator to use.
    - `trace`: trace-based interval generator
    - `poisson`: poisson interval generator
    - `gamma`: gamma interval generator
    - `static`: static interval generator

for trace-based interval generator,

- `trace_request_interval_generator_config_seed`: seed for the random number generator.
- `trace_request_interval_generator_config_trace_file`: path to the trace file.
    - `./data/processed_traces`
- `trace_request_interval_generator_config_start_time`: start time of the trace file.
- `trace_request_interval_generator_config_end_time`: end time of the trace file.
- `trace_request_interval_generator_config_time_scale_factor`: scale factor for the time.

for poisson interval generator,

- `poisson_request_interval_generator_config_seed`: seed for the random number generator.
- `poisson_request_interval_generator_config_qps`: Query Per Second (QPS) of the request generator.

for gamma interval generator,

- `gamma_request_interval_generator_config_seed`: seed for the random number generator.
- `gamma_request_interval_generator_config_qps`: Query Per Second (QPS) of the request generator.
- `gamma_request_interval_generator_config_cv`: coefficient of variation (CV) of the request generator.

for static interval generator,

- `static_request_interval_generator_config_seed`: seed for the random number generator.

### trace replay request generator

- `trace_replay_request_generator_config_seed`: seed for the random number generator.
- `trace_replay_request_generator_config_trace_file`: path to the trace file.
    - `./data/processed_traces`
- `trace_replay_request_generator_config_max_tokens`: maximum number of tokens in a request.
- `trace_replay_request_generator_config_prefill_scale_factor`: scale factor for the prefill requests.
- `trace_replay_request_generator_config_decode_scale_factor`: scale factor for the decode requests.
- `trace_replay_request_generator_config_time_scale_factor`: scale factor for the time.

## execution time predictor

- `execution_time_predictor_config_type`: type of execution time predictor to use.
    - `linear_regression`: linear regression execution time predictor, based on sklearn.linear_model.LinearRegression
    - `random_forrest`: random forest execution time predictor, based on sklearn.ensemble.RandomForestRegressor
    - `dummy`: dummy execution time predictor, always predicts 0.


Some important parameters for linear regression execution time predictor:

- `linear_regression_execution_time_predictor_config_polynomial_degree`: degree of the polynomial to use for the linear regression model.
- `linear_regression_execution_time_predictor_config_include_bias`: whether to include the bias term in the linear regression model.
- `linear_regression_execution_time_predictor_config_polynomial_interaction_only`: whether to only include polynomial interactions in the linear regression model.
- `linear_regression_execution_time_predictor_config_fit_intercept`: whether to fit the intercept term in the linear regression model.

for random forrest execution time predictor,

- `random_forrest_execution_time_predictor_config_num_estimators`: number of trees in the random forest.
- `random_forrest_execution_time_predictor_config_max_depth`: maximum depth of the trees in the random forest.
- `random_forrest_execution_time_predictor_config_min_samples_split`: minimum number of samples required to split an internal node in the random forest.

## capacity management tips

Each scheduler has different memory management and batch formation strategies. Here are some tips for capacity management:

- vLLM scheduler and Sarathi scheduler have better capacity management implementations. wrong setup is okay.
- the rest of the schedulers are not as good at capacity management. wrong setup will lead to runtime errors.

one can set `batch_size_cap` to a small number to reduce the risk of runtime errors.

- for ORCA scheduler, batch_size_cap <= (prediction_max_prefill_per_request / prefill_tokens) ** 2

- for FasterTransformer scheduler,

- for LightLLM scheduler,