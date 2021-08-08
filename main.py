import time
import json
from step.init_step import init_step
from step.drop_dup_step import drop_dup_step
from step.recall_step import recall_step
from step.predict_step import predict_step
from step.info_show_step import info_show_step
from utils.context import Context


def request(uid):
    f = open('params.json', 'r')
    steps = json.load(f)['steps']
    f.close()

    print(steps)

    ctx = Context(uid)
    for step in steps:
        DefinedClass = eval(step + "_step")
        DefinedObject = DefinedClass()
        DefinedObject.process(ctx)
        print(f"step {step},num of groups: {len(ctx.groups)}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='input uid to request')
    parser.add_argument('--uid', type=int, default=1, help='uid')
    args = parser.parse_args()

    uid = args.uid
    print(uid)

    request(uid = uid)


