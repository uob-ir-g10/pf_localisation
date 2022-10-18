from geometry_msgs.msg import Pose, PoseArray, Quaternion, PoseWithCovarianceStamped, Point
from . pf_base import PFLocaliserBase
import math
import rospy

from . util import rotateQuaternion, getHeading
from random import gauss, choices

from time import time


class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters

        # These values will be used in  PFLocaliserBase.predict_from_odometry()
        self.ODOM_ROTATION_NOISE = 0        # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0     # Odometry model x-axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0           # Odometry model y-axis (side-to-side) noise

        self.SAMPLE_ROTATION_NOISE = 0.05		
        self.SAMPLE_TRANSLATION_NOISE = 0.025

        self.INITIAL_ROTATION_NOISE = 0.25
        self.INITIAL_TRANSLATION_NOISE = 0.2

        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        
       
    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """
        particle_cloud = PoseArray()
        if isinstance(initialpose, PoseWithCovarianceStamped):
            initialpose = initialpose.pose.pose
        for i in range(self.NUMBER_PREDICTED_READINGS):
            generated_particle = self.add_initial_noise(initialpose)            
            particle_cloud.poses.append(generated_particle)

        return particle_cloud

 
    
    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

        """
        # Generate weighted list of particles
        weights = []
        particles = self.particlecloud.poses.copy()
        for particle in particles:
            weights.append(self.sensor_model.get_weight(scan, particle))

        # Sample from the weighted list while adding resampling noise        
        self.particlecloud.poses = list(map(self.add_sample_noise, choices(particles, weights=weights, k=self.NUMBER_PREDICTED_READINGS)))

    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
        """

        # Calculate mean for all points, throw away half that are furthest, and recalculate mean. Should be alright if points are not outrageously far away
        # as should be the case considering gaussian (= unlikely)
        estimate = Pose()

        positions = list(map(lambda pose: pose.position, self.particlecloud.poses))
        orientations = list(map(lambda pose: pose.orientation, self.particlecloud.poses))


        # POSITION ESTIMATION
        pos_mean = get_mean(positions)
        distances = []
        for position in positions:
            distance = get_distance_squared(position, pos_mean)
            distances.append((position, distance))
            
        # Sort positions in descending order based on distance to cluster center
        distances.sort(key=lambda i: i[1], reverse=True)

        # Only keep the points closest to center
        points = list(map(lambda x: x[0], distances[:len(distances)//2]))
        pos_mean_no_outliers = get_mean(points)

        estimate.position = Point(*pos_mean_no_outliers)

        # ORIENTATION ESTIMATION
        ori_mean = get_mean(orientations)
        distances = []
        for quat in orientations:
            distance = get_distance_squared(quat, ori_mean)
            distances.append((quat, distance))
        
        # Sort orientations in descending order based on distance to cluster center
        distances.sort(key=lambda i: i[1], reverse=True)
        points = list(map(lambda x: x[0], distances[:len(distances)//2]))
        ori_mean_no_outliers = get_mean(points)

        estimate.orientation = Quaternion(*ori_mean_no_outliers)

        return estimate

    def add_sample_noise(self, pose: Pose):
        return self.add_noise(pose, rotation_noise=self.SAMPLE_ROTATION_NOISE, translation_noise=self.SAMPLE_TRANSLATION_NOISE)
    
    def add_initial_noise(self, pose: Pose):
        return self.add_noise(pose, rotation_noise=self.INITIAL_ROTATION_NOISE, translation_noise=self.INITIAL_TRANSLATION_NOISE)

    def add_noise(self, pose: Pose, rotation_noise, translation_noise):
        """
        Adds sampling noise to a particle

        :Args:
            | pose: the original pose
            | rotation_noise: noise given to the rotation of the pose
            | translation_noise: noise given to the translation of the pose
        :Return:
            | (geometry_msgs.msg.Pose) pose with added sampling noise
        """
        noisy = Pose()
        noisy.orientation = rotateQuaternion(pose.orientation, gauss(0, rotation_noise))
        noisy.position.x = gauss(pose.position.x, translation_noise) 
        noisy.position.y = gauss(pose.position.y, translation_noise) 
        noisy.position.z = 0
        return noisy
            
def to_iter(msg):
    """
    Convert a ros msg object into an iterator over its slots

    :Args:
        | msg: a ros msg object
    :Return:
        | a generator over the object's slots 
    """
    # Don't do anything if msg is already iterable
    if hasattr(msg, '__iter__'):
        for x in msg:
            yield x
    # Otherwise iterate over the msg's attributes
    else:
        slots = msg.__slots__
        for slot in slots:
            yield(getattr(msg, slot))


def get_mean(points):
    """
    Returns the mean value of an array of `Pose`s/`Orientation`s
    """
    mean = [0.0] * len(list(to_iter(points[0])))
    for point in points:
        for i, x in enumerate(to_iter(point)):
            mean[i] += x
    for i in range(len(mean)):
        mean[i] /= len(points)
    return mean

def get_distance_squared(p1, p2):
    """
    Get the distance squared between two `Pose`s/`Orientation`s
    """
    return sum(map(lambda a,b: (a-b)**2, to_iter(p1), to_iter(p2)))