from detectron2.engine import DefaultTrainer as d2Trainer
from config import get_cfg
import fsod
def main():
    cfg = get_cfg()
    cfg.merge_from_file("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/config/fct.yaml")
    cfg.freeze()
    model = d2Trainer.build_model(cfg)
    print(model)
    
if __name__ == "__main__":
    main()