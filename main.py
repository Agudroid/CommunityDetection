import argparse

parser = argparse.ArgumentParser(description="Community Detection app")

parser.add_argument('--csv', type=str, help='ruta del csv a leer')
parser.add_argument('--type', type=str, help='random, shortest-path')