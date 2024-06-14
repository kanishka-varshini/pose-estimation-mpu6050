import math 
import rclpy
import smbus2
import numpy as np
from time import sleep
from rclpy.node import Node
from geometry_msgs.msg import Pose
from ahrs.filters import Madgwick


#mpu6050 node to publish the mpu data
#output msg format is String
class MPU6050Publisher(Node):
    def __init__(self):
        super().__init__('mpu6050_publisher')
        self.publisher_=self.create_publisher(Pose, 'pose',10)
        timer_period=1.0
        self.timer=self.create_timer(timer_period, self.timer_callback)
        self.bus=smbus2.SMBus(1)
        self.MPU6050_ADDR = 0x68
        self.PWR_MGMT_1 = 0x6B
        self.ACCEL_XOUT_H = 0x3B
        self.ACCEL_YOUT_H = 0x3D
        self.ACCEL_ZOUT_H = 0x3F
        self.GYRO_XOUT_H = 0x43
        self.GYRO_YOUT_H = 0x45
        self.GYRO_ZOUT_H = 0x47

        self.ACCL_SCALE_FACTOR = 16384.0
        self.GRAVITY = 9.81
        self.GYRO_SCALE_FACTOR =131.0
        
        self.timestep=0.01

        
        self.velocity=np.zeros(3)
        self.position=np.zeros(3)

        self.quaternions=[]
        self.quaternions.append(np.array([1.0,0.0,0.0,0.0]))
        self.count=0
        self.madgwick = Madgwick(sampleperiod=self.timestep)


        self.bus.write_byte_data(self.MPU6050_ADDR, self.PWR_MGMT_1, 0)


    def read_raw_data(self,addr):
        high = self.bus.read_byte_data(self.MPU6050_ADDR, addr)
        low = self.bus.read_byte_data(self.MPU6050_ADDR, addr + 1)
        value = ((high << 8) | low)
        if value > 32768:
            value -= 65536
        return value

    def timer_callback(self):
        self.count+=1
        accl_scale=self.GRAVITY/self.ACCL_SCALE_FACTOR
        gyro_scale= math.pi/(180*self.GYRO_SCALE_FACTOR) #gyro data to radians/second
        # the imu gives raw data in terms of g, multiply by scale factor and divide by 9.81 to get accl in m/s^2
        # gyro value is also in raw data units, to convert to deg/s divide by sensitivity scale factor
        acc_x = self.read_raw_data(self.ACCEL_XOUT_H)*accl_scale
        acc_y = self.read_raw_data(self.ACCEL_YOUT_H)*accl_scale
        acc_z = self.read_raw_data(self.ACCEL_ZOUT_H)*accl_scale
        gyro_x = self.read_raw_data(self.GYRO_XOUT_H)*gyro_scale
        gyro_y = self.read_raw_data(self.GYRO_YOUT_H)*gyro_scale
        gyro_z = self.read_raw_data(self.GYRO_ZOUT_H)*gyro_scale

        acc=np.array([acc_x,acc_y,acc_z])
        
        #since ros2 uses quaternions to represent the orientation, using madgwick filter to convert the data to the quaternion form
        
        self.quaternions.append(self.madgwick.updateIMU(self.quaternions[self.count-1],np.array([gyro_x,gyro_y,gyro_z]),acc))

        #calculate the velocity to estimate the position

        #converting accelerations to world frame
        acc_world=self.rotate_vector_by_quaternion(acc,self.quaternions[self.count])

        #subtract gravity - not relevant here because robot motion is assumed to be planar
        acc_world-=np.array([0.0,0.0,9.81])

        #update vel an dposition
        dt=self.timestep
        self.velocity+=acc_world*dt
        self.position+=self.velocity*dt

        pose_msg=Pose()

        pose_msg.position.x= self.position[0]
        pose_msg.position.y= self.position[1]
        pose_msg.position.z= self.position[2]

        q=self.quaternions[self.count]
        pose_msg.orientation.w=q[0]
        pose_msg.orientation.x=q[1]
        pose_msg.orientation.y=q[2]
        pose_msg.orientation.z=q[3]


        self.publisher_.publish(pose_msg)
        self.get_logger().info(f"Publishing: {pose_msg}")

    def rotate_vector_by_quaternion(self, v, q):
        
        q_conj = [q[0], -q[1], -q[2], -q[3]]
        v_quat = [0.0, v[0], v[1], v[2]]
        rotated_v = self.quaternion_multiply(self.quaternion_multiply(q, v_quat), q_conj)
        return np.array([rotated_v[1], rotated_v[2], rotated_v[3]])

    def quaternion_multiply(self, q1, q2):
        
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return [w, x, y, z]



def main(args=None):
    rclpy.init(args=args)
    mpu6050_publisher = MPU6050Publisher()
    rclpy.spin(mpu6050_publisher)
    mpu6050_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
