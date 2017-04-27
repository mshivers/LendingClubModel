import constants
import invest

def main():
    model_dir = constants.PathManager.get_dir('training') 
    pm = invest.PortfolioManager(model_dir=model_dir, required_return=0.09)
    pm.try_for_awhile(1)


if __name__=='__main__':
    main()

