from tokenize import Pointfloat
from geometry_msgs.msg import Pose, PoseArray, Quaternion, PoseWithCovarianceStamped, Point
from . pf_base import PFLocaliserBase
import math
import rospy

from . util import rotateQuaternion, getHeading
from random import gauss, choices, random, choice
from sklearn.cluster import DBSCAN

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

        # ----- Particle parameters
        self.STANDARD_PARTICLES = 100           # Number of particles in particle cloud
        self.KIDNAPPED_PARTICLES_LOW = 50       # Number of randomly generated particles to combat kidnapping
        self.KIDNAPPED_PARTICLES_HIGH = 100
        self.KIDNAPPED_PARTICLES = self.KIDNAPPED_PARTICLES_HIGH

        self.possible_positions = [] # list of indices of occupancy grid data[] that correspond to valid robot positions. recalculated each time occupancy map is recieved      

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
        # Generate a list of possible indices where (random) particles can reside
        self.possible_positions = [i for i, x in enumerate(self.occupancy_map.data) if x != -1 and x < 50]

        # Generate the particle cloud
        particle_cloud = PoseArray()
        if isinstance(initialpose, PoseWithCovarianceStamped):
            initialpose = initialpose.pose.pose
        for i in range(self.STANDARD_PARTICLES):
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
        # Add random particles to help with kidnapped robot problem
        for i in range(self.KIDNAPPED_PARTICLES):
            particles.append(self.get_random_particle())
        weights = [self.sensor_model.get_weight(scan, particle) for particle in particles]

        # Sample from the weighted list while adding resampling noise        
        self.particlecloud.poses = list(map(self.add_sample_noise, choices(particles, weights=weights, k=self.STANDARD_PARTICLES)))
        


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
        estimate = Pose()

        positions = list(map(lambda pose: [pose.position.x, pose.position.y], self.particlecloud.poses))
        orientations = list(map(lambda pose: pose.orientation, self.particlecloud.poses))

        clustering = DBSCAN(
            # max distance between two samples for one to be considered as in the neighbourhood of the other
            eps=self.SAMPLE_TRANSLATION_NOISE*2, # 95% of standard particles are generated within 2 sigma
            # number of samples in a nieghbourhood to be considerd a core point
            min_samples=5
        ).fit(positions)

        if len(clustering.components_) > 0:
            poses = [(positions[i], orientations[i]) for i,x in enumerate(clustering.labels_) if x == 0]
            estimate.position = Point(*[*get_mean([pos for pos, ori in poses]), 0.0])
            estimate.orientation = Quaternion(*get_mean([ori for pos, ori in poses]))
            self.KIDNAPPED_PARTICLES = self.KIDNAPPED_PARTICLES_LOW
        else:
            # If dbscan finds no cluster, need to find some point that works. Maybe at this point we increase # of random particles?
            estimate.position = Point(*[*positions[0], 0.0])
            estimate.orientation = orientations[0]
            self.KIDNAPPED_PARTICLES = self.KIDNAPPED_PARTICLES_HIGH

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

    def get_random_particle(self):
        """
        Return a random particle that's atleast on the map somewhere
        """
        position = choice(self.possible_positions)
        p = Pose()
        p.position.x = position % self.occupancy_map.info.width * self.occupancy_map.info.resolution
        p.position.y = position // self.occupancy_map.info.height * self.occupancy_map.info.resolution
        p.position.z = 0
        p.orientation = rotateQuaternion(Quaternion(w=1.0), random() * 2 * math.pi)
        return p

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
