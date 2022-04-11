import torch
import random
import numpy as np
import os

from config_util import cli_main
from utilities import createLoader_split_train_val_hospitalization, eval_model_hospitalization, set_seed

from models.mytransformer import Transformer, Trainer_Hospitalization

def main():
    args = cli_main()

    print('args.model')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    args.device = device 
    set_seed(args)


    # ------------
    # data
    # ------------
   
    if args.split_train_val:
        train_loader, val_loader, test_loader, test_state_loader, state_dict, county_dict = createLoader_split_train_val_hospitalization(args)
    else:
        train_loader,test_loader, test_state_loader, state_dict, county_dict = createLoader_train_test_hospitalization(args)
            
    
    if args.model == "transformer":
        model = Transformer(args)

    print(model)
    # ------------
    # Training
    # ------------
    trainer = Trainer_Hospitalization(args)
   
    if args.split_train_val:
        trainer.fit(model, train_loader, val_loader, test_state_loader)
    

    # ------------
    # Testing
    # ------------
    if args.split_train_val:
        final_model_path = args.model_dir + '/last.ckpt'
               
        final_model = Transformer(args)
        checkpoint = torch.load(final_model_path)
        final_model.load_state_dict(checkpoint['model_state_dict'])

        eval_model_hospitalization(args, final_model, device, test_state_loader, final_model_path, state_dict, county_dict, description="final model state prediction")
        
if __name__ == '__main__':
    main()
 
