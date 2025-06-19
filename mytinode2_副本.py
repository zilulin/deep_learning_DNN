import numpy as np
import matplotlib.pyplot as plt
import time
# 向量投影函数
# 计算向量 a 在向量 b 上的投影
def vector_projection(a, b):

    b_norm_sq = np.dot(b, b)
    if b_norm_sq < 1e-8:
        raise ValueError("基向量 b 的模长不能为零")
    projection = np.dot(a, b) / b_norm_sq * b
    return projection

#预处理及初始化
class set_rope_node:
    def __init__(self,node_number,top,end,rope_length,line_mass):
        self.node_number = node_number
        self.top = top
        self.end = end
        self.rope_length = rope_length
        self.lien_mass = line_mass
        top = np.array(top)
        end = np.array(end)
        self.pos_of_each_node = self.calculate_node_positions(top,end,node_number)
        self.each_line_length = self.calculate_each_line_length(rope_length,node_number)
        self.node_mass = self.calculate_node_mass(line_mass,node_number)
        self.velcity = self.calculate_line_velcity(node_number)

        #计算绳子中间节点位置的函数
    def calculate_node_positions(self, top, end, node_number):
        #定义空数组，初始化
        hole_node = np.zeros((node_number-1,2,3),dtype = float)
        middle_node = np.zeros((node_number-2,3), dtype = float)

        #计算差值
        direction = end - top
        for i in range (node_number-2):
            middle_node[i,:] = top + direction / (node_number-1) * (i+1)
        hole_node[0,0,:] = top
        hole_node[-1,1,:] = end
        for i in range(node_number-2):
            hole_node[i+1,0,:] = middle_node[i,:]
            hole_node[i,1,:] = middle_node[i,:]

        return hole_node
    #计算每一段的长度
    def calculate_each_line_length(self,rope_length,node_number):
        each_length = np.zeros(node_number-1)
        each_line_length = rope_length/(node_number - 1)
        for i in range(node_number-1):
            each_length[i] = each_line_length
        return each_length
    
    #计算每个节点的质量
    def calculate_node_mass(self,line_mass,node_number):
        node_mass = np.zeros(self.node_number, dtype=float)
        # 计算每个节点的质量
        for i in range(node_number):
            if i == 0 or i == node_number -1 :
                node_mass[i] = 0.5 * line_mass / (self.node_number-1)
            else:
                # 中间节点质量为总质量的平均分配
                node_mass[i] = line_mass / (self.node_number - 1)
        return node_mass
    
    def calculate_line_velcity(self,node_number):
        velcity = np.zeros((node_number - 1,2,3),dtype = float)
        return velcity
        
class hanging_multinode:
    #绳子基本参数
    def __init__(self,pos,vel,line_length,node_mass,node_number,d,EA,C,gravity = 9.8):
        #基础变量赋值
        self.pos = pos
        self.vel = vel
        self.line_length = line_length
        self.node_mass = node_mass
        self.node_number = node_number
        self.d = d
        self.EA = EA
        self.C = C
        self.gravity = gravity
        #定义新的数组，用来存放数据,计算合力
        self.tension = self.calculate_tension_force(pos,line_length,node_number,d,EA)
        self.damping = self.calculate_damping_force(pos,vel,line_length,node_number,d,C)
        self.global_force = self.calculate_global_force(node_mass,gravity,node_number)
        self.total_force = self.tension + self.damping + self.global_force
    
    def calculate_tension_force(self,pos,line_length,node_number,d,EA):
        force = np.zeros((node_number - 1,2,3))
        for i in range (node_number-1):
            sward_before = pos[i,0,:]- pos[i,1,:]
            length_now = np.linalg.norm(sward_before)
            if length_now <=1e-6:
                force [i,:,:] = 0
                continue
            sward = sward_before/length_now
            length_changetate = length_now - line_length[i]
            if length_changetate <= 0 :
                force [i,:,:] = 0
                continue
            force[i,1,:] =  1/4*np.pi*EA*d**2 * length_changetate * sward
            force[i,0,:] = -1/4*np.pi*EA*d**2 * length_changetate * sward
        return force
    
    def calculate_damping_force(self,pos,vel,line_length,node_number,d,C):
        force = np.zeros((node_number - 1,2,3),dtype = float)
        epsilon = 0
        for i in range (node_number-1):
            #rel_pos = np.zeros(3)
            rel_pos = pos[i,0,:]- pos[i,1,:]
            rel_vel = vel[i,0,:]- vel[i,1,:]
            #print("rel",rel_pos,rel_vel)
            if np.isnan(np.linalg.norm(rel_pos)*line_length[i]) == 1: 
                force [i,1,:] = 0
                continue
            epsilon = np.dot (rel_pos,rel_vel)  /  (np.linalg.norm(rel_pos)*line_length[i])
            # print(rel_vel)
            # print(epsilon)
            #单位向量
            sward = rel_pos/np.linalg.norm(rel_pos)

            #阻尼   
            force [i,1,:] = C * np.pi * 0.25 * d**2 * epsilon * sward
            force [i,0,:] = - force [i,1,:]
            
        return force
    
    #重力
    def calculate_global_force(self,node_mass,gravity,node_number):
        force = np.zeros((node_number - 1,2,3),dtype=float)
        g_acc = np.array([0, 0,  -float(gravity)], dtype=float)
        #将重力作用在每个节点上
        #采取将节点的重量视作集中点的一半，在顶端节点和尾端节点都施加重力
        for i in range(self.node_number - 1):
            if i == 0:
                force[i,0,:]= g_acc * node_mass[i]  # 顶端节点受力
                force[i,1,:] = g_acc * node_mass[i+1] * 0.5 # 尾端节点受力
                #print("Global force calculated for first node:", force[0:3, 0, i], "and second node:", force[0:3, 1, i])
            elif i == self.node_number - 2:
                force[i,0,:] = g_acc * node_mass[i]*0.5
                force[i,1,:]  = g_acc * node_mass[i+1]
                #print("Global force calculated for last node:", force[0:3, 0, i], "and second last node:", force[0:3, 1, i])
            else:
                force[i,0,:] = g_acc * node_mass[i] * 0.5
                force[i,1,:]  = g_acc * node_mass[i+1] * 0.5
                #print("Global force calculated for node", i, ":", force[0:3, 0, i], "and node", i+1, ":", force[0:3, 1, i])
        #print("Global force calculated:", force)
        return force
    
class total_acc:
    def __init__(self,total_force,total_mass,node_number):
        self.total_force  = total_force
        self.total_mass = total_mass
        self.node_number = node_number
        self.acc = self.calculate_acc(total_force,total_mass,node_number)
        self.acc_trance = self.calculate_acc_tarnce(self.acc,node_number)
    
    def calculate_acc(self,total_force,total_mass,node_number):
        acc = np.zeros((node_number,3))
        for i in range(node_number):
            if i == 0:
                acc [i,:] = total_force[i,0,:]/total_mass[i]
                continue
            if i == node_number - 1:
                acc [i,:] = total_force[i-1,1,:]/total_mass[i]
                continue
            acc [i,:] = (total_force[i,0,:] + total_force[i-1,1,:])/total_mass[i]
        return acc
    def calculate_acc_tarnce(self,acc,node_number):
        acc_trance = np.zeros((node_number - 1,2,3),dtype=float)
        for i in range(node_number-1):
            # if i == 0:
            #     acc_trance[i,0,:] = acc[i,:]
            #     continue
            # if i == node_number - 2:
            #     acc_trance[i,1,:] = acc[i+1,:]
            #     continue
            # acc_trance[i,0,:] = acc_trance[i-1,1,:] = acc [i,:]
            acc_trance[i,0,:] = acc[i,:]
            acc_trance[i,1,:] = acc[i+1,:]

            #端点不移动：
        acc_trance[0,0,:] = 0.0
    

        return acc_trance
    
def acc_func(pos ,vel ,each_line_length,node_mass,node_number,d,EA,C,gravity = 9.8):
    rope_line = hanging_multinode(pos ,vel ,each_line_length,node_mass,node_number,d,EA,C,gravity)
    rope_acc = total_acc(rope_line.total_force,node_mass,node_number)
    
    return rope_acc.acc_trance
def rk4(pos ,vel ,each_line_length,node_mass,node_number,d,EA,C,t0,h,gravity):
    pos_down= pos  # 备份位移
    vel1 = vel2 = vel3 = vel
    t = t0
    ##龙格库塔四阶求速度
    k1 = acc_func(pos ,vel ,each_line_length,node_mass,node_number,d,EA,C,gravity = 9.8)
    vel1 = vel + k1 * 0.5* h
    #print(k1)
    #pos = pos_down + vel1 * h + k1 * h**2 * 0.5
    k2 = h * acc_func(pos ,vel1,each_line_length,node_mass,node_number,d,EA,C,gravity = 9.8)
    vel2 = vel + k2 * 0.5 * h
    #pos = pos_down + vel2 * h + k2 * h**2 * 0.5
    k3 = h * acc_func(pos ,vel2 ,each_line_length,node_mass,node_number,d,EA,C,gravity = 9.8)
    vel3 = vel + k3 * h
    #pos = pos_down + vel3 * h + k3 * h**2 * 0.5
    k4 = h * acc_func(pos ,vel3 ,each_line_length,node_mass,node_number,d,EA,C,gravity = 9.8)

    pos = pos_down + vel * h + (k1 + 2*k2 + 2*k3 + k4) / 6.0 * h*0.5 * h
    vel = vel+(k1 + 2*k2 + 2*k3 + k4) / 6.0 *h

    t = t+h
    print(t)
    return pos,vel,t


if __name__ == "__main__":

#主程序
    start_time = time.time()

    plt.rcParams["font.family"] = ["SimHei", "SimHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False            # 解决负号 '-' 显示为方块的问题

    top = [0,0,0]
    end = [0,0,-61.5]
    rope_length = 60
    node_number = 20
    rope = set_rope_node(node_number,top,end,rope_length,line_mass=8000)
    d = 0.05
    EA = 1000000000
    C = 50000000
    #rope.velcity[2,1,:] =np.array([0,0,-1]) 
    #rope.node_mass[-1] = 70700

    # rope_line = hanging_multinode(rope.pos_of_each_node ,rope.velcity ,rope.each_line_length,rope.node_mass,node_number,d,EA,C,gravity = 9.8)
    # rope_acc = total_acc(rope_line.total_force,rope.node_mass,node_number)
    # acc = rope_acc.acc

    # 用于保存数据
    time_list = []
    top_z_list = []
    bottom_z_list = []


    t = 0

    while t < 20 :



        if t > 20 :
        #     rope.node_mass[-1] = 70700
            C = 500000000



    # 记录当前时刻和端点位移（这里记录 z 坐标，pos[2]为上端，pos[5]为下端）
        # time_list.append(t)
        # top_z_list.append(rope.pos_of_each_node[-1,1,2])
        # bottom_z_list.append(rope.pos_of_each_node[0,0,2])

        rope.pos_of_each_node,rope.velcity,t = rk4(rope.pos_of_each_node ,rope.velcity ,rope.each_line_length,rope.node_mass,node_number,d,EA,C,t,h=0.001,gravity = 9.8)
        #rope.pos_of_each_node[0,0,:] = np.array([0,0,0])
        


        #绞车

    #找到平衡位置之后开始计算
    #速度归零
    rope.velcity = np.zeros((node_number-1,2,3))
    #时间归零
    t = 0
    #调整阻尼系数加快稳定
    C = 700000000
    #开始循环
    while t < 20 :

    #增加吊物
        #rope.node_mass[-1] = 7070

    



    # 记录当前时刻和端点位移（这里记录 z 坐标，pos[2]为上端，pos[5]为下端）
        time_list.append(t)
        
        #bottom_z_list.append(rope.pos_of_each_node[0,0,2])

        rope.pos_of_each_node,rope.velcity,t = rk4(rope.pos_of_each_node ,rope.velcity ,rope.each_line_length,rope.node_mass,node_number,d,EA,C,t,h=0.001,gravity = 9.8)
        #rope.pos_of_each_node[0,0,:] = np.array([0,0,0])
        #计算拉力
        rope_line_calculate_force = hanging_multinode(rope.pos_of_each_node ,rope.velcity ,rope.each_line_length,rope.node_mass,node_number,d,EA,C,gravity = 9.8)
        rope_line_calculate_force.total_force

        if t >  10 :
            rope.pos_of_each_node[0,0,2]+=0.001
        
        bottom_z_list.append(rope.pos_of_each_node[-1,0,2])
        top_z_list.append(rope_line_calculate_force.total_force[0,1,2])

    end_time = time.time()
    print("calculate_time:",-(start_time - end_time))

    plt.figure(figsize=(10, 6))
    plt.plot(time_list, bottom_z_list, label="下端点 z 坐标", linestyle="-",color = "k")
    
    plt.xlabel("时间 (s)")
    plt.ylabel("位移 (m)")
    plt.title("绳子端点位移随时间变化")
    plt.legend()


    plt.figure(2)
    plt.plot(time_list, top_z_list, label="上端点 z 拉力", linestyle="-")
    plt.xlabel("时间 (s)")
    plt.ylabel("拉力 (m)")
    plt.title("绳子拉力随时间变化")
    plt.legend()
    plt.show()

    #print("position\n",rope.pos_of_each_node)
    # print("length\n",rope.each_line_length)
    print("nodemass\n",rope.node_mass)
    # print("node_velcity\n",rope.velcity)
    # print ("tension\n",rope_line.tension)
    # print ("damping\n",rope_line.damping)
    # print("g",rope_line.global_force)
    #print("total_force",rope_line.total_force)