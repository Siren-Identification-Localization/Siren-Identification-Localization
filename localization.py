import math
import numpy as np
import scipy.io.wavfile
from argparse import ArgumentParser

# sound travels 343 m/s
sound_velocity = 343

# put the two neighbor edges of the angle you are trying to calculate
def cosine_rule(e1, e2, e3):
    return (e1*e1 + e2*e2 - e3*e3) / (2*e1*e2)

# solves the quadratic equation (ax^2 + bx + c = 0)
def get_distance(a, b, c):
    # Get distances from quadratic formula: not sure when a < 0
    if a < 0:
        x = (-b - np.sqrt(b*b-4*a*c)) / (2*a)
    else:
        x = (-b + np.sqrt(b*b-4*a*c)) / (2*a)
    return x

# solves the quadratic equation (ax^2 + bx + c = 0)
# where b is even number
def get_distance_prime(a, b, c):
    b_prime = b/2
    if a < 0:
        x = (-b_prime - np.sqrt(b_prime*b_prime-a*c)) / a
    else:
        x = (-b_prime + np.sqrt(b_prime*b_prime-a*c)) / a
    return x

def find_peak(x):
    for i in range(x.shape[0]):
        prev = x[i]
        curr = x[i+1]
        if curr < prev:
            break
    return i

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i1', '--infile1', help="Input wave for microphone1") 
    parser.add_argument('-i2', '--infile2', help="Input wave for microphone2") 
    parser.add_argument('-i3', '--infile3', help="Input wave for microphone3") 
    parser.add_argument('-d12', '--d12', type=float, default=None, help='d1 - d2')
    parser.add_argument('-d31', '--d31', type=float, default=None, help='d3 - d1')
    parser.add_argument('-R12', '--R12', type=float, default=None, help='R1 - R2')
    parser.add_argument('-R23', '--R23', type=float, default=None, help='R2 - R3')
    parser.add_argument('-R31', '--R31', type=float, default=None, help='R3 - R1')
    args = parser.parse_args()

    sampling_rate1, data1 = scipy.io.wavfile.read(args.infile1)
    sampling_rate2, data2 = scipy.io.wavfile.read(args.infile2)
    sampling_rate3, data3 = scipy.io.wavfile.read(args.infile3)

    # find the first peak
    peak1 = find_peak(data1)
    peak2 = find_peak(data2)
    peak3 = find_peak(data3)

    d12 = float(peak1 - peak2) * sound_velocity / sampling_rate1
    d31 = float(peak3 - peak1) * sound_velocity / sampling_rate1
    

    # d12 = args.d12
    # d31 = args.d31
    d23 = -1 * (d12 + d31)
    R12 = args.R12
    R23 = args.R23 
    R31 = args.R31 

    ########################################################################
    # Part2: Get distances from the source and three microphones using distance differences
    ########################################################################

    # Calculate coefficients
    a = 4*d12*d23*(R12*R12+R23*R23-R31*R31) \
            + 4*(d12*d12*R23*R23+d23*d23*R12*R12) \
            - 2*(R12*R12*R23*R23+R23*R23*R31*R31+R31*R31*R12*R12) \
            + pow(R12,4)+pow(R23,4)+pow(R31,4)
    
    b = -2*(d12*d23*d23*(R12*R12+R23*R23-R31*R31)) \
            - d12*(R12*R12*R23*R23+R23*R23*R31*R31-pow(R23,4)) \
            + d12*d12*d23*(R12*R12+R23*R23-R31*R31) \
            + d23*R12*R12*(R23*R23+R31*R31-R12*R12) \
            + 2*(pow(d12,3)*R23*R23-pow(d23,3)*R12*R12)

    c = -1*(d12*d12*d23*d23*(R12*R12+R23*R23-R31*R31)) \
            - d12*d12*R23*R23*(R12*R12+R31*R31-R23*R23) \
            - d23*d23*R12*R12*(R23*R23+R31*R31-R12*R12) + pow(d12,4)*R23*R23 \
            + pow(d23,3)*d23*R12*R12 + R12*R12*R23*R23*R31*R31

    # Get distances from quadratic formula
    d2 = get_distance_prime(a, b, c)

    d1 = d2 + d12
    d3 = d2 - d23

    # print("d1: {:0.3f}, d2: {:0.3f}, d3: {:0.3f}".format(d1, d2, d3))

    # Calculate the angles
    cosTheta12 = cosine_rule(d1, d2, R23)
    cosTheta23 = cosine_rule(d2, d3, R23)
    cosTheta31 = cosine_rule(d3, d1, R31)

    theta12 = math.degrees(math.acos(cosTheta12))
    theta23 = math.degrees(math.acos(cosTheta23))
    theta31 = math.degrees(math.acos(cosTheta31))

    # print("Theta12: {:0.3f}, Theta23: {:0.3f}, Theta31: {:0.3f}".format(theta12, theta23, theta31))

    ########################################################################
    # Part2: Get the distance and angle from the circumcenter and the source    
    ########################################################################

    # Calculate the radius of circumcenter
    cosTheta3 = cosine_rule(R31, R23, R12)
    sinTheta3 = np.sqrt(1-cosTheta3*cosTheta3)
    r = R12/(2*sinTheta3)

    a = R12*R12
    b = (-d1*d1-d2*d2-2*r*r+R12*R12)*R12*R12
    c = (R12*R12-2*r*r)*d1*d1*d2*d2 + (r*r-d1*d1-d2*d2)*R12*R12*r*r + (pow(d1,4)+pow(d2,4))*r*r

    x = get_distance(a, b, c)

    d0 = np.sqrt(x)

    theta012 = math.degrees(math.acos(cosine_rule(r, R12, r)))
    theta10A = math.degrees(math.acos(cosine_rule(r, d0, d1)))
    theta0 = 180 - theta012 - theta10A

    print("d0: {:0.3f}, theta0: {:0.3f}degrees".format(d0, theta0))
