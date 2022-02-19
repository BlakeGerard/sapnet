from train import *
from sapnet import SAPNetActorCritic
from os.path import exists

load_path = "model/goddard.pt"

def main():
	model = SAPNetActorCritic("goddard")
	if (exists(load_path)):
		model.load(load_path)
	
	trainer = ActorCriticTrainer(model)
	trainer.train()

if __name__ == '__main__':
	main()
