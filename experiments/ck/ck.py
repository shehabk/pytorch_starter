from __future__ import print_function
import sys
import torch
## Should be run from the project root
# project_dir = os.getcwd()
## For debug purpose have the actual path
project_dir = '/media/shehabk/D_DRIVE/codes/code_practice/pytorch_starter'
sys.path.insert( 0 , project_dir )
from configs.base_config import BaseClassificationOptions
from agents.base_classification_agent import BaseClassicationAgent

def main():
    
    config = BaseClassificationOptions().parse()
    agent = BaseClassicationAgent(config)
    agent.run()



if __name__ == '__main__':
    main()