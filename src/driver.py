from train import *
from server import *
from sapnet import SAPNetActorCritic
from os.path import exists

load_path = "models/goddard.old.pt"
role = Role.HOST

def main():
	model = SAPNetActorCritic("goddard")
	if (exists(load_path)):
		print("Loading goddard") 
		model.load(load_path)

	trainer = ActorCriticTrainer(model, role)
	trainer.train()

if __name__ == '__main__':
	main()
