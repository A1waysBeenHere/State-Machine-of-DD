from vllm import LLM, SamplingParams

# 1. 加载模型
llm = LLM(model="/workspace/mnt/transfer_folder/models/Qwen2.5-32B", dtype="bfloat16", tensor_parallel_size=4, gpu_memory_utilization=0.8)

# 2. 设置推理参数
sampling_params = SamplingParams(
    temperature=0.6, top_p=0.95, max_tokens=1024, n=1
)

# 3. 运行推理
# prompt = r'''The circle $x^2 + y^2 = 2$ and the parabola $y^2 = 8x$ have two common tangents, forming four points of tangency. Find the area of the quadrilateral formed by the four points of tangency.

# [asy]
# unitsize(0.8 cm);

# real upperparab (real x) {
# return (sqrt(8*x));
# }

# real lowerparab (real x) {
# return (-sqrt(8*x));
# }

# pair A, B, C, D;

# A = (-1,1);
# B = (2,4);
# C = (-1,-1);
# D = (2,-4);

# draw(graph(upperparab,0,3));
# draw(graph(lowerparab,0,3));
# draw(Circle((0,0),sqrt(2)));
# draw(interp(A,B,-0.2)--interp(A,B,1.2));
# draw(interp(C,D,-0.2)--interp(C,D,1.2));
# draw(A--C);
# draw(B--D);

# dot(A);
# dot(B);
# dot(C);
# dot(D);
# [/asy]'''

input_tokens = [[151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 641, 279, 13549, 11, 3040, 25362, 315, 10578, 220, 16, 448, 35182, 400, 47, 54876, 400, 48, 54876, 400, 49, 54876, 323, 400, 50, 3, 525, 68660, 311, 825, 2441, 323, 311, 279, 11067, 315, 57960, 55114, 19360, 54876, 438, 6839, 13, 508, 6405, 921, 2141, 7, 17, 15, 15, 317, 12670, 362, 11, 425, 11, 356, 11, 393, 11, 1207, 11, 431, 11, 328, 280, 49, 4539, 15, 11, 15, 317, 48, 63242, 17, 11, 15, 317, 50, 4539, 17, 11, 15, 317, 47, 4539, 16, 11, 16, 13, 22, 18, 17, 317, 33, 63242, 20, 13, 22, 18, 4999, 16, 317, 34, 4539, 18, 13, 22, 18, 17, 4999, 16, 317, 32, 4539, 16, 13, 18, 21, 21, 11, 18, 13, 15, 24, 23, 317, 7633, 4346, 313, 33, 313, 34, 313, 32, 317, 7633, 86154, 5304, 11, 220, 16, 1106, 7633, 86154, 6253, 11, 220, 16, 1106, 7633, 86154, 2785, 11, 220, 16, 1106, 7633, 86154, 3759, 11, 220, 16, 1106, 1502, 445, 32, 497, 362, 11, 451, 317, 1502, 445, 33, 497, 425, 11, 13387, 317, 1502, 445, 34, 497, 356, 11, 5052, 317, 16119, 5304, 317, 16119, 6253, 317, 16119, 2785, 317, 16119, 3759, 317, 1502, 445, 47, 497, 393, 11, 451, 317, 1502, 445, 48, 497, 1207, 11, 13387, 317, 1502, 445, 49, 497, 431, 11, 13387, 317, 1502, 445, 50, 497, 328, 11, 5052, 317, 24157, 6405, 21675, 3838, 374, 279, 8381, 6629, 315, 279, 24632, 9210, 304, 21495, 400, 47, 70810, 3, 30, 151645, 198, 151644, 77091, 198]] * 500

output = llm.generate(prompt_token_ids=input_tokens, sampling_params=sampling_params, use_tqdm=True)

# # 4. 打印结果
# for i, out in enumerate(output[0].outputs):
#     print(f"Sample {i+1}:\n {out.text}\n\n")

print(output)
