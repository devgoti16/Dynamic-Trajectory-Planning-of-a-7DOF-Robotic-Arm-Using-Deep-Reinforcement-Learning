#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

def publish_joint_positions():
    rospy.init_node('joint_position_publisher', anonymous=True)
    #-------------------------------------
    # pub = rospy.Publisher('/j2s7s300/joint_states', JointState, queue_size=10)
    #rate = rospy.Rate(10)  # 10hz
    # joint_state = JointState()
    # joint_state.name = ['j2s7s300_joint_1', 'j2s7s300_joint_2', 'j2s7s300_joint_3', 'j2s7s300_joint_4', 'j2s7s300_joint_5', 'j2s7s300_joint_6', 'j2s7s300_joint_7']
    # joint_state.position = [0.5, 1.0, -0.5, 1.0, 0.0, -1.0, 0.5]
    # joint_state.velocity = []
    # joint_state.effort = []
    # while not rospy.is_shutdown():
    #     joint_state.header.stamp = rospy.Time.now()
    #     rospy.loginfo(joint_state)
    #     pub.publish(joint_state)
    #     rate.sleep()
    #----------------------------------------------------------

    #      this makes the arm joint to have particular angle
    #pub = rospy.Publisher('/j2s7s300/joint_4_position_controller/command',Float64, queue_size= 10)
    rate = rospy.Rate(10)  # 10hz
    # data = Float64()
    # data = -3.14
    # for i in range(40):
    #     rospy.loginfo(data)
    #     pub.publish(data)
    #     data = data + 0.15
    #     rate.sleep()
    #--------------------

         #this gets the value of veolcity and angles from gazebo
    # def callbackf(msg):
    #     angles = msg.position
    #     velocity = msg.velocity
    #     print(velocity.type)

    #     print('angles')
    #     print(angles)
    #     print("velocity")
    #     print(velocity)


    # sub = rospy.Subscriber('/j2s7s300/joint_states', JointState, callbackf)
    # rospy.spin()

    # pub1 = rospy.Publisher('/j2s7s300/joint_2_velocity_controller/command',Float64, queue_size= 10)
    # pub2 = rospy.Publisher('/j2s7s300/joint_2_velocity_controller/command',Float64, queue_size= 10)
    # pub3 = rospy.Publisher('/j2s7s300/joint_2_velocity_controller/command',Float64, queue_size= 10)
    pub4 = rospy.Publisher('/j2s7s300/joint_4_velocity_controller/command',Float64, queue_size= 10)
    # pub5 = rospy.Publisher('/j2s7s300/joint_2_velocity_controller/command',Float64, queue_size= 10)
    # pub6 = rospy.Publisher('/j2s7s300/joint_2_velocity_controller/command',Float64, queue_size= 10)
    # pub7 = rospy.Publisher('/j2s7s300/joint_2_velocity_controller/command',Float64, queue_size= 10)
    rate = rospy.Rate(10)  # 10hz
    data = Float64()
    data = 0.0
    # for i in range(40):
    #     rospy.loginfo(data)
    #     pub.publish(data)
    #     pub2.publish(data)
    #     data = data + 0.02
    #     rate.sleep()
    # for i in range(40):
    #     rospy.loginfo(data)
    #     pub.publish(data)
    #     pub2.publish(data)
    #     data = data - 0.02
    #     rate.sleep()
    # pub1.publish(data)
    # pub2.publish(data)
    # pub3.publish(data)
    pub4.publish(data)
    # pub5.publish(data)
    # pub6.publish(data)
    # pub7.publish(data)
    # while not rospy.is_shutdown():
    #     rospy.loginfo(data)
    #     pub1.publish(data)
    #     pub2.publish(data)
    #     pub3.publish(data)
    #     pub4.publish(data)
    #     pub5.publish(data)
    #     pub6.publish(data)
    #     pub7.publish(data)

    





    


if __name__ == '__main__':
    try:
        publish_joint_positions()
    except rospy.ROSInterruptException:
        pass
