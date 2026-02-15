#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.base_model import BaseModel

def main():
    config_path = 'config/config.yaml'
    base_model = BaseModel(config_path)
    X, y = base_model.generate_data()
    base_model.train(X, y)
    base_model.save()
    print("Base model trained and saved.")

if __name__ == "__main__":
    main()
