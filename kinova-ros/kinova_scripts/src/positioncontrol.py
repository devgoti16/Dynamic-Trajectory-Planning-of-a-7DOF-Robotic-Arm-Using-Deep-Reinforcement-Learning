import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np
from scipy.optimize import minimize

class Jaco2PositionController:
    def __init__(self):
        rospy.init_node('positioncontrol')
        self.joint_names = [
            'joint_1', 'joint_2', 'joint_3',
            'joint_4', 'joint_5', 'joint_6', 'joint_7'
        ]
        self.joint_angle_pubs = [
            rospy.Publisher(f'/j2s7s300/{name}_position_controller/command', Float64, queue_size=10)
            for name in self.joint_names
        ]
        self.target_point = np.array([3.0, 2.5, 3.2])  # Example target point
        self.current_joint_states = None
        rospy.Subscriber('/j2s7s300/joint_states', JointState, self.joint_state_callback)
        self.rate = rospy.Rate(10)  # 10 Hz
        self.loginfo('Control started')

        self.d_parameters = [0.2755,0.2050,0.2050,0.2073,0.1038,0.1038,0.1600,0.0098] #d1,d2,d3,d4,d5,d6,d7,e2  
        

    def forward_kinematics(self, joint_angles):
        # Placeholder for forward kinematics calculation
        # This function should return the current end-effector position
        # based on the provided joint angles.

        dh_parameters = [
            (np.radians(90), 0, -self.d_parameters[0], joint_angles[0]),
            (np.radians(90), 0, 0,  joint_angles[1]),
            (np.radians(90), 0, -(self.d_parameters[1]+self.d_parameters[2]),  joint_angles[2]),
            (np.radians(90), 0, -self.d_parameters[7],  joint_angles[3]),
            (np.radians(90), 0, -(self.d_parameters[3]+self.d_parameters[4]), joint_angles[4]),
            (np.radians(90), 0, 0,  joint_angles[5]),
            (np.radians(180), 0, -(self.d_parameters[5]+self.d_parameters[6]),  joint_angles[6])
        ] #alpha,d,a,thetea
        T_0_n = np.eye(4)
        transformation = []
        for i, (alpha,d, a, theta) in enumerate(dh_parameters):
            T_i = self.dh_transformation(alpha, d, a, theta)
            T_0_n = np.dot(T_0_n, T_i)
            transformation.append(T_0_n)
        return T_0_n
    
    def dh_transformation(alpha, d, a, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
    

    def joint_state_callback(self, msg):
        self.current_joint_states = msg

    
    def compute_position(self):
        # Ensure the current joint states are available
        if self.current_joint_states is None:
            return np.zeros(len(self.joint_names))

        joint_angles = np.array(self.current_joint_states.position)
        T__0 = self.forward_kinematics(joint_angles)
        current_position = T__0[:3,3]
        return current_position
    
    def calculate_jacobian(joint_angles, dh_params):
        n = len(joint_angles)
        J = np.zeros((3,7))
        T = np.eye(4)
        pos_0  = np.array([0,0,0])
        z_0 = np.array([0,0,1])

        for i in range(7):
            theta = joint_angles[i]
            T_i = dh_transformation(dh_params[i])
            T = np.dot(T,T_i)
            pos_i = T[:3,3]
            z_i = T[:3,2]

            J[:,i] = np.cross(z_i, (self.compute_position(joint_angles)- pos_i))

        return J

    
    def control_loop(self):
        tolerance  = 1e-6
        while not rospy.is_shutdown():
            if self.current_joint_states is not None:

                current_position = self.compute_position()
                error = np.linalg.norm(np.array(self.target_point) - np.array(current_position))
                # gradient = current_position - self.target_point
                # joint_angles_publish = self.current_joint_states.position - 0.05*gradient
                # self.joint_angle_pubs[:].publish(joint_angles_publish)
                rospy.loginfo(joint_angles_publish)
                if(error<tolerance):
                    break

                J = self.calculate_jacobian(joint_angles)
                    
                self.rate.sleep()

if __name__ == '__main__':
    controller = Jaco2PositionController()
    controller.control_loop()
