import math
import numpy as np
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d12', '--d12', type=float, default=None, help='d1 - d2')
    parser.add_argument('-d31', '--d31', type=float, default=None, help='d3 - d1')
    parser.add_argument('-R12', '--R12', type=float, default=None, help='R1 - R2')
    parser.add_argument('-R23', '--R23', type=float, default=None, help='R2 - R3')
    parser.add_argument('-R31', '--R31', type=float, default=None, help='R3 - R1')
    args = parser.parse_args()

    d12 = args.d12
    d31 = args.d31
    d23 = -1 * (d12 + d31)
    R12 = args.R12
    R23 = args.R23 
    R31 = args.R31 

    # Calculate coefficients
    a = 4*d12*d23*(pow(R12,2)+pow(R23,2)-pow(R31,2)) \
            + 4*(pow(d12,2)*pow(R23,2)+pow(d23,2)*pow(R12,2)) \
            - 2*(pow(R12,2)*pow(R23,2)+pow(R23,2)*pow(R31,2)+pow(R31,2)*pow(R12,2)) \
            + pow(R12,4)+pow(R23,4)+pow(R31,4)
    
    b = -1*(d12*pow(d23,2)*(pow(R12,2)+pow(R23,2)-pow(R31,2))) \
            - d12*pow(R23,2)*(pow(R12,2)+pow(R31,2))\
            + d23*pow(R12,2)*(pow(R23,2)+pow(R31,2))\
            + pow(d12,2)*d23*(pow(R12,2)+pow(R23,2)-pow(R31,2)) \
            + d12*pow(R23,4)-d23*pow(R12,4)\
            + 2*(pow(d12,3)*pow(R23,2)-pow(d23,3)*pow(R12,2))

    c = -1*(pow(d12,2)*pow(d23,2)*(pow(R12,2)+pow(R23,2)-pow(R31,2))) \
            - pow(d12,2)*pow(R23,2)*(pow(R12,2)+pow(R31,2)-pow(R23,2)) \
            - pow(d23,2)*pow(R12,2)*(pow(R23,2)+pow(R31,2)-pow(R12,2)-pow(d23,2)) \
            + pow(R23,2)*(pow(d12,4)+pow(R12,2)*pow(R31,2)) 

    # a = 4*d12*d23*(R12*R12+R23*R23-R31*R31) \
            # + 4*(d12*d12*R23*R23+d23*d23*R12*R12) \
            # - 2*(R12*R12*R23*R23+R23*R23*R31*R31+R31*R31*R12*R12) \
            # + pow(R12,4)+pow(R23,4)+pow(R31,4)
    
    # b = -1*(d12*d23*d23*(R12*R12+R23*R23-R31*R31)) \
            # - d12*(R12*R12*R23*R23+R23*R23*R31*R31-pow(R23,4)) \
            # + d12*d12*d23*(R12*R12+R23*R23-R31*R31) \
            # + d23*R12*R12*(R23*R23+R31*R31-R12*R12) \
            # + 2*(pow(d12,3)*R23*R23-pow(d23,3)*R12*R12)

    # c = -1*(d12*d12*d23*d23*(R12*R12+R23*R23-R31*R31)) \
            # - d12*d12*R23*R23*(R12*R12+R31*R31-R23*R23) \
            # - d23*d23*R12*R12*(R23*R23+R31*R31-R12*R12) + pow(d12,4)*R23*R23 \
            # + pow(d23,3)*d23*R12*R12 + R12*R12*R23*R23*R31*R31

    # Get distances from quadratic formula
    if a < 0:
        d2 = (-b - np.sqrt(b*b-a*c)) / a
    else:
        d2 = (-b + np.sqrt(b*b-a*c)) / a

    d1 = d2 + d12
    d3 = d2 - d23

    print("d1: {:0.3f}, d2: {:0.3f}, d3: {:0.3f}".format(d1, d2, d3))

    # Calculate the angles
    cosTheta12 = (d1*d1 + d2*d2 - R12*R12) / (2*d1*d2)
    cosTheta23 = (d2*d2 + d3*d3 - R23*R23) / (2*d2*d3)
    cosTheta31 = (d3*d3 + d1*d1 - R31*R31) / (2*d3*d1)

    theta12 = math.degrees(math.acos(cosTheta12))
    theta23 = math.degrees(math.acos(cosTheta23))
    theta31 = math.degrees(math.acos(cosTheta31))

    print("Theta12: {:0.3f}, Theta23: {:0.3f}, Theta31: {:0.3f}".format(theta12, theta23, theta31))

