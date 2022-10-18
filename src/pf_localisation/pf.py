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

        self.SAMPLE_ROTATION_NOISE = 0.1		
        self.SAMPLE_TRANSLATION_NOISE = 0.1	

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
            generated_particle = self.add_noise(initialpose)            
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
        self.particlecloud.poses.clear()
        for i in range(self.NUMBER_PREDICTED_READINGS):
            sample = choices(particles, weights=weights, k=1)[0]
            self.particlecloud.poses.append(self.add_noise(sample))

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


        def to_iter(msg):
            """
            Convert ros msg object to iterator over it's slots
            """
            if hasattr(msg, '__iter__'):
                return msg.__iter__
            slots = msg.__slots__
            for slot in slots:
                yield(getattr(msg, slot))


        def get_mean(points):
            mean = [0.0] * len(list(to_iter(points[0])))
            for point in points:
                for i, x in enumerate(to_iter(point)):
                    mean[i] += x
            for i in range(len(mean)):
                mean[i] /= len(points)
            return mean
        
        def get_distance_squared(p1, p2):
            return sum(map(lambda a,b: (a-b)**2, to_iter(p1), to_iter(p2)))


        # Calculate mean for all points, throw away half that are furthest, and recalculate mean. Should be alright if points are not outrageously far away
        # as should be the case considering gaussian (= unlikely)
        estimate = Pose()

        positions = list(map(lambda pose: pose.position, self.particlecloud.poses))
        orientations = list(map(lambda pose: pose.orientation, self.particlecloud.poses))


        pos_mean = get_mean(positions)

        # rospy.loginfo(positions)
        # distances = []
        # for position in positions:
        #     distance = get_distance_squared(position, pos_mean)
        #     #distance = (position.x - pos_mean[0])**2 + (position.y - pos_mean[1])**2 + (position.z - pos_mean[2])**2
        #     distances.append((position, distance))
        # # Sort poses in descending order based on distance to cluster center

        # # all distances are 0 for some reason btw
        # distances.sort(key=lambda i: i[1], reverse=True)

        # # Should remove the distances in the points array before calling get_mean as the mean becomes '[]'
        # points = distances[:len(distances)//2]
        # rospy.loginfo(points)
        # pos_mean_no_outliers = get_mean(points)


        # estimate.position = Point(*pos_mean_no_outliers)
        estimate.position = Point(*pos_mean)

        ori_mean = get_mean(orientations)
        # distances = []
        # for quat in orientations:
        #     distance = get_distance_squared(quat, ori_mean)
        #     distances.append((quat, distance))
        # distances.sort(key=lambda i: i[1], reverse=True)
        # points = distances[:len(distances)//2]
        # ori_mean_no_outliers = get_mean(points)

        # estimate.orientation = Quaternion(*ori_mean_no_outliers)
        estimate.orientation = Quaternion(*ori_mean)

        return estimate

    def add_noise(self, pose: Pose):
        """
        Adds sampling noise to a particle

        :Args:
            | pose: the original pose
        :Return:
            | (geometry_msgs.msg.Pose) pose with added sampling noise
        """
        noisy = Pose()
        noisy.orientation = rotateQuaternion(pose.orientation, gauss(0, self.SAMPLE_ROTATION_NOISE))
        noisy.position.x = gauss(pose.position.x, self.SAMPLE_TRANSLATION_NOISE) 
        noisy.position.y = gauss(pose.position.y, self.SAMPLE_TRANSLATION_NOISE) 
        noisy.position.z = 0
        return noisy
            
