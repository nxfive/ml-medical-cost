import hydra

from .stage import run


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
