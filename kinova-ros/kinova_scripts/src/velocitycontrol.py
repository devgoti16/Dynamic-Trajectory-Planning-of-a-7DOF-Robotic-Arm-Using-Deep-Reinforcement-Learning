import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np

class Jaco2VelocityController:
    def __init__(self):
        rospy.init_node('velocitycontrol')
        self.joint_names = [
            'joint_1', 'joint_2', 'joint_3',
            'joint_4', 'joint_5', 'joint_6', 'joint_7'
        ]
        self.joint_velocity_pubs = [
            rospy.Publisher(f'/j2s7s300/{name}_velocity_controller/command', Float64, queue_size=10)
            for name in self.joint_names
        ]
        self.target_point = np.array([3.0, 2.5, 3.2])  # Example target point
        self.current_joint_states = None
        rospy.Subscriber('/j2s7s300/joint_states', JointState, self.joint_state_callback)
        self.rate = rospy.Rate(10)  # 10 Hz

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
        ]
        T_0_n = np.eye(4)
        for i, (alpha,d, a, theta) in enumerate(dh_parameters):
            T_i = self.dh_transformation(alpha, d, a, theta)
            T_0_n = np.dot(T_0_n, T_i)

        end_effector_position = T_0_n[:3, 3]
        return end_effector_position
    
    def dh_transformation(alpha, d, a, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
    

    def compute_jacobian(self, joint_angles):
        # Placeholder for Jacobian matrix calculation
        # This function should return the Jacobian matrix based on the provided joint angles.
        J = np.eye(3, len(joint_angles))  # Example placeholder
        return J

    def joint_state_callback(self, msg):
        self.current_joint_states = msg

    
    def compute_joint_velocities(self):
        # Ensure the current joint states are available
        if self.current_joint_states is None:
            return np.zeros(len(self.joint_names))

        joint_angles = np.array(self.current_joint_states.position)
        current_position = self.forward_kinematics(joint_angles)
        error = self.target_point - current_position

        J = self.compute_jacobian(joint_angles)

        # Use the pseudoinverse of the Jacobian to compute joint velocities
        J_pinv = np.linalg.pinv(J)
        joint_velocities = J_pinv.dot(error)

        return joint_velocities

    def control_loop(self):
        while not rospy.is_shutdown():
            if self.current_joint_states is not None:
                joint_velocities = self.compute_joint_velocities()
                for pub, vel in zip(self.joint_velocity_pubs, joint_velocities):
                    pub.publish(Float64(vel))
            self.rate.sleep()

if __name__ == '__main__':
    controller = Jaco2VelocityController()
    controller.control_loop()
