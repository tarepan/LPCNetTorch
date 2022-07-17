"""Run the inference"""


import argparse
import torch

from .model import Model


if __name__ == "__main__":  # pragma: no cover

    parser = argparse.ArgumentParser(description='Run Inference')
    parser.add_argument("-m", "--model-ckpt-path", required=True)
    parser.add_argument("-i", "--i-path",          required=True)
    parser.add_argument("-o", "--o-path",          required=True)
    parser.add_argument("--device")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model: Model = Model.load_from_checkpoint(checkpoint_path=args.model_ckpt_path).to(device) # type: ignore ; because of PyTorch Lightning
    model.eval()

    with torch.inference_mode():
        raw = model.load(args.i_path)
        batch = model.preprocess(raw, args.device)
        o_pred = model.predict_step(batch, batch_idx=0)

    # Tensor[Batch=1, ...] => Tensor[...] => NDArray[...]
    o_wave = o_pred[0].to('cpu').detach().numpy()

    # Output
    print(o_wave)
    ## Audio
    # sf.write()
