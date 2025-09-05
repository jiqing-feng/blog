---
title: "Optimizing GPT OSS on 6th Gen Xeon at GCP"
thumbnail: /blog/assets/xxx.png # waiting for upload
authors:
- user: Jiqing
  guest: true
  org: Intel
- user: MatrixYao
  guest: true
  org: Intel
- user: kding1
  guest: true
  org: Intel
---


# Optimizing GPT OSS on 6th Gen Xeon at GCP

With our optimization, GPT-OSS achieves inference speeds that approach human reading speed up to batch size 4. We have merged all optimizations into Transformers (PR [40304](https://github.com/huggingface/transformers/pull/40304) and [40545](https://github.com/huggingface/transformers/pull/40545)), so users can benefit from them out of the box. Users can apply GNR powered C4 and setup environments by following [huggingface.co/blog/intel-gcp-c4](https://huggingface.co/blog/intel-gcp-c4) to get the results in this blog.


## Introduction

GPT OSS is an open-weight model known for its strong reasoning and versatility. Its MoE architecture, while having a large number of parameters, activates only a small subset during inference. This makes it possible to run large models on Intel Xeon CPUs, where Expert Parallelism can further improve performance by distributing the computation of experts across multiple processes.

In this blog, we benchmark the bfloat16 version of GPT OSS-20B ([lmsys/GPT OSS-20b-bf16](https://huggingface.co/lmsys/GPT OSS-20b-bf16)) on Intel 6th Gen Xeon GNR CPUs at GCP C4. Our results demonstrate that GPT OSS model can reach human reading speed on text generation tasks. Additionally, EP improves TPOT by 5%–70% as batch size increases, enabling significantly higher throughput compared to non-parallel setups.


## Expert Parallelism

Expert Parallelism is a technique used to distribute the computation of experts across multiple NUMA nodes. In the GPT OSS model, the experts are evenly split and assigned to different NUMA nodes on Intel Xeon CPUs. By leveraging EP, the model can achieve significant speed-ups, especially for large-scale MoE architectures.

<kbd>
  <img src="assets/GPT OSS-on-intel-xeon/expert_parallelism.png">
</kbd>

To enable EP on GPT OSS model in transformers, we just need to pass `tp_plan="auto"` when loading the model. This is because the experts performs grouped_gemm strategy in the GPT OSS [_tp_plan](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/configuration_gpt_oss.py#L40-L45).

We recommend you to use the command `mpirun -np 2 --map-by ppr:1:numa --bind-to numa` to bind the computation of different experts to different NUMA nodes. This ensures that each NUMA node handles its assigned experts independently, leveraging the locality of memory and computation to improve performance and reduce communication overhead.

`mpirun -np 2 --map-by ppr:1:numa --bind-to numa -genv MASTER_ADDR=127.0.0.1 -genv MASTER_PORT=29500 -genv OMP_NUM_THREADS=<cores_per_numa> python tp_hf.py`

```diff
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# The mpirun use PMI_RANK and PMI_SIZE as default env to pass rank and world size.
# We need to set RANK, LOCAL_RANK and WORLD_SIZE which can be recognized by transformers.
os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['LOCAL_RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

model_id = "lmsys/GPT OSS-20b-bf16"
# Load model with tp_plan="auto" to enable Expert Parallelism
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
+    tp_plan="auto"
)
# Prepare input tokens
tokenizer = AutoTokenizer.from_pretrained(model_id)
messages = [
    {"role": "user", "content": "How many rs are in the word 'strawberry'?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```


## Performance Evaluation of Expert Parallelism

To evaluate the performance of Expert Parallelism (EP), we fixed the input and output sequence lengths to 1024 tokens and tested the latency under different batch sizes. The tests were conducted on Intel 6-th Gen Xeon GNR CPU powered GCP C4 with and without EP enabled.

The following figures show the performance results for TTFT (Time to First Token) and TPOT (Time per Output Token) under various batch sizes.

<kbd>
  <img src="assets/GPT OSS-on-intel-xeon/TTFT-GPT OSS.png">
</kbd>

<kbd>
  <img src="assets/GPT OSS-on-intel-xeon/TPOT-GPT OSS.png">
</kbd>

In the TTFT results, we observed that EP could complete the prefill in 1 second for batch size 1. However, as batch size increases, EP becomes slower than non-EP. This is because non-EP utilizes all the computational resources available on the instance, while EP limits each group of experts to only a portion of the resources. Non-EP benefits from more resources, resulting in better TTFT performance at larger batch sizes.

In the TPOT results, we observed that both EP and non-EP configurations can achieve human reading speed. Human reading speed is 240~300ms per word, so we can achieve that up to batch size 4. Moreover, EP demonstrates better performance as batch size increases. By distributing expert computation across multiple NUMA nodes, EP allows each node to process its workload independently. This reduces the computational burden on a single process and improves overall efficiency. Additionally, EP leverages memory and computation locality within each NUMA node, minimizing communication overhead and achieving better scalability. Ultimately, EP achieved a throughput of 95 tokens per second when batch size is 64.


## Conclusion

This blog demonstrates the potential of running large MoE models on CPUs. With further optimizations, we look forward to unlocking even greater performance on CPU-based systems in the future.
