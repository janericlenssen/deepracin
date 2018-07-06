import numpy as np

rmax = 32-1
x0 = 32
y0 = 32
m_pi_f = 3.14159265358979323846

m = 0


def main():
    CcodeXc = "int sradxc[{rmax}][180] = {".replace("{rmax}", str(rmax+1))

    CcodeYc = "int sradyc[{rmax}][180] = {".replace("{rmax}", str(rmax+1))

    # generate indices for srad up to 64 x 64
    #print(m)
    xc = np.zeros(shape=(rmax+1, 180))
    yc = np.zeros(shape=(rmax+1, 180))

    for r in range(2, rmax+1):

        # insert zeros
        if r == 2:
            CcodeXc += "{"
            CcodeYc += "{"
            for i in range(2):
                for j in range(180):
                    CcodeXc += "0,"
                    CcodeYc += "0,"
                    if j == 179:
                        CcodeXc = CcodeXc[:-1]
                        CcodeYc = CcodeYc[:-1]
                CcodeXc += "}, {"
                CcodeYc += "}, {"

            #if i == 1:
            #    CcodeXc = CcodeXc[:-2]
            #    CcodeYc = CcodeYc[:-2]
        else:
            CcodeXc += "{"
            CcodeYc += "{"


        for angle in range (91,271):
            array_index = angle-91
            theta = angle*(m_pi_f/180)
            xc[r][array_index] = int(round(r*np.cos(theta)) + x0)
            yc[r][array_index] = int(round(r*np.sin(theta)) + y0)

            CcodeXc += "{xc},".replace("{xc}", str(int(xc[r][array_index])))

            CcodeYc += "{yc},".replace("{yc}", str(int(yc[r][array_index])))

            if array_index == 179:
                CcodeXc = CcodeXc[:-1]
                CcodeYc = CcodeYc[:-1]

        CcodeXc += "}, "
        CcodeYc += "}, "
        if r == rmax:
            CcodeXc = CcodeXc[:-2]
            CcodeYc = CcodeYc[:-2]

    CcodeXc += "};"
    CcodeYc += "};"


    '''for i in range(180):
        print(i)
        for j in range(180):
            print(j)
            print(xc[180*i + j], yc[180*i +j])
        print("next")'''

    print(CcodeXc)
    print(CcodeYc)

if __name__=="__main__":
    main()
