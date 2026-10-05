import argparse
import torch
from common.benchmark import measure, save_measurements
from common.records import run_directory
from lab1.attention import attention as explicit, reference as sdpa
from lab2.attention.attention import attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation",choices=["explicit","sdpa","streaming"],required=True)
    parser.add_argument("--split",choices=["dev","heldout"],default="dev")
    parser.add_argument("--block-q",type=int,choices=[16,32],default=16)
    parser.add_argument("--block-k",type=int,choices=[32,64],default=32)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    seed=42 if args.split=="dev" else 2026
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("highest")
    shapes=[256,512,1024] if args.split=="dev" else [383,1537]
    fn={"explicit":explicit,"sdpa":sdpa,
        "streaming":lambda q,k,v:attention(q,k,v,args.block_q,args.block_k)}[args.implementation]
    rows=[]
    with run_directory(args.out,{**vars(args),"seed":seed,"B":1,"H":2,"D":64,
                                "S":shapes,"dtype":"float32","device":"cuda",
                                "warmup":20,"iterations":100,"repeats":3}) as out:
        for s in shapes:
            inputs=tuple(torch.randn(1,2,s,64,device="cuda") for _ in range(3))
            for row in measure(fn,inputs,sdpa,"cuda"):
                row.update(shape=f"S{s}",S=s,implementation=args.implementation,
                           block_q=args.block_q,block_k=args.block_k,split=args.split)
                rows.append(row)
            save_measurements(out,rows)


if __name__=="__main__":
    main()

