import constants
import invest

def main():
    model_dir = constants.PathManager.get_dir('training') 
    pm = invest.PortfolioManager(model_dir=model_dir, new_only=False)
    pm.try_for_awhile(0)


if __name__=='__main__':
    main()

