import os
import shutil
import cx_Oracle


def read_cxdata(app_id):
    try:
        conn = cx_Oracle.connect('ecom/passwd111@192.168.0.188/orcl188')  # 连接数据库
        c = conn.cursor()  # 获取cursor
        sql = ''' select ALGORITHM_EN_NAME from  APP_CONFIG_RECORD where APP_ID IN {}'''.format(tuple(app_id))
        # print(sql)
        x = c.execute(sql)  # 使用cursor进行各种操作
        rows = x.fetchall()
        c.close()  # 关闭cursor
        conn.close()  # 关闭连接
        return rows
    except Exception as e:
        print("请输出正确的APP_ID", e)


def copy_model_name(name_path, model_name):
    try:
        path = '/data2/gztpai/project/afw_ai_engine/algorithm/{}/'.format(name_path)
        copypath = '/data2/gztpai/modelfile/{}/'.format(name_path)

        for i in model_name:
            print(path + i[0] + '======copy>>>>>' + copypath + i[0])
            shutil.copytree(path + i[0], copypath + i[0])
    except Exception:
        print('发现错误，查看模型名是不是错了？')


if __name__ == '__main__':
    modelname = input("请输入模型APP_ID(用','隔开)：")
    # modelname = "1801f14ef65739688d55bba3a1432395,0c35ea8293d4317b8185ec01e4b9e58d,4b80f88580ec3aa09b3d51965b31ba87"
    onename = modelname.split(',')

    namelist = read_cxdata(app_id=onename)
    copy_model_name(name_path='code', model_name=namelist)
    copy_model_name(name_path='model', model_name=namelist)
