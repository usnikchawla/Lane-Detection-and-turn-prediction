import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    frame = cv2.VideoCapture('challenge.mp4')

    while (frame.isOpened()):
        success, image = frame.read()
        if not success:
            print("\nEnd of frames\n")
            break

        # image = cv2.flip(image, 1)
        rows, cols, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        region = np.array([(230, rows - 60), (1120, rows - 60), (730, 440), (610, 440)])
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, pts=[region], color=(255, 255))
        cropped = cv2.bitwise_and(gray, mask)

        flat_plane = np.array([(0, 0), (200, 0), (200, 500), (0, 500)], dtype=float)
        h, status = cv2.findHomography(region, flat_plane)
        h_inv, status_inv = cv2.findHomography(flat_plane,region)
        warped = cv2.warpPerspective(cropped, h, (200,500))
        warped = cv2.flip(warped,0)
        dst = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        # dst = warped
        blur = cv2.GaussianBlur(warped, (3,3),0)
        edge = cv2.Canny(warped, 50, 450)  # 50,350

        linesP = cv2.HoughLinesP(edge, 2, np.pi / 180, 25, np.array([]), minLineLength=10, maxLineGap=150) # image, rho, theta, threshold = min # of intersections for line, lines, min, max

        # segregate left lines from right lines
        left = []
        right = []

        # get 8 best fit points from each line
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                if l[0] < 100 and l[2] < 100:
                    left.append(l)
                    # cv2.line(dst, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    right.append(l)
                    # cv2.line(dst, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 3, cv2.LINE_AA)

        # segregating into points falling on the left or right curves and visualizing the points
        all_points_left = []
        for i in left:
            all_points_left.append((i[0],i[1]))
            all_points_left.append((i[2], i[3]))

        # for i in all_points_left:
        #     cv2.circle(dst, i, 5, (255,0,0), 1)

        all_points_right = []
        for i in right:
            all_points_right.append((i[0],i[1]))
            all_points_right.append((i[2], i[3]))

        # for i in all_points_right:
        #     cv2.circle(dst, i, 5, (255,0,0), 1)

        # check if detection can be satisfactorily performed
        # p1 = (0,50)
        # p2 = (0,200)
        # p3 = (50,200)
        # p4 = (50,50)
        #
        # dst = cv2.line(dst, p1, p2, (255,0,50), 3)
        # dst = cv2.line(dst, p2, p3, (255, 0, 50), 3)
        # dst = cv2.line(dst, p3, p4, (255, 0, 50), 3)
        # dst = cv2.line(dst, p4, p1, (255, 0, 50), 3)

        count = 0
        for i in all_points_left:
            x = i[0]
            y = i[1]

            if x > 0 and x < 50 and y > 50 and y < 200:
                count = count + 1

        # start polynomial fitting only if detection is possible
        if count > 0:
            # fit a polynomial
            print("Lane detected.")
            left = np.array(all_points_left)
            for i in range(len(left)):
                x_l = left[:, 0]
                y_l = left[:, 1]
            right = np.array(all_points_right)
            for i in range(len(right)):
                x_r = right[:, 0]
                y_r = right[:, 1]

            left_fit = np.polyfit(y_l, x_l, 2)
            right_fit = np.polyfit(y_r, x_r, 2)

            # print("New Frame")
            # print("Left curve coefficients:" + str(left_fit))
            # print("Right curve coefficients:" + str(right_fit) + "\n")

            ploty = np.linspace(0, dst.shape[0] - 1, dst.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            leftcurve = np.int_(np.vstack((left_fitx, ploty)).T)
            rightcurve = np.int_(np.vstack((right_fitx, ploty)).T)

            cv2.polylines(dst, [leftcurve], False, (0, 255, 0),5)
            cv2.polylines(dst, [rightcurve], False, (0, 0, 255),5)
        else:
            # else previous best fit will be plotted
            print("Lane detection failed! Resorting to previous best lane.")
            ploty = np.linspace(0, dst.shape[0] - 1, dst.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            leftcurve = np.int_(np.vstack((left_fitx, ploty)).T)
            rightcurve = np.int_(np.vstack((right_fitx, ploty)).T)

            cv2.polylines(dst, [leftcurve], False, (0, 255, 0),5)
            cv2.polylines(dst, [rightcurve], False, (0, 0, 255),5)

        # # testing with matplotlib for individual frames
        # plt.imshow(dst)
        # plt.plot(left_fitx,ploty,color = 'blue')
        # plt.plot(right_fitx, ploty, color='black')
        # plt.show()

        # find radius of curvature
        left_curvature = ((1+(2*left_fit[0]*x_r[0]+left_fit[1])**2)**1.5)/np.absolute(2*left_fit[0])
        right_curvature = ((1+(2*right_fit[0]*x_r[0]+right_fit[1])**2)**1.5)/np.absolute(2*right_fit[0])
        road_curvature = (left_curvature + right_curvature) / 2
        
        info1 = ("Radius of Curvature: "+str(road_curvature)+" m ")
        print(info1)

        # detect left/right turn
        # if derivative is negative then it is a right turn
        slope = 2 * right_fit[0]*5+right_fit[1]
        slope_left = 2 * left_fit[0] * 5 + left_fit[1]
        print(slope)
        print(slope_left)
        if slope < 0:
            info2 = ("Turn: Right")
            print(info2)
            print("\n")
        else:
            info2 = ("Turn: Left")
            print(info2)
            print("\n")


        # inverse warp the warped image
        final = cv2.flip(dst,0)
        final = cv2.warpPerspective(final, h_inv, (1280, 720))
        # cv2.fillConvexPoly(image, pts_dst.astype(int), 0, 16);

        # superimposing detected lane onto the original image
        final = cv2.add(image, final)

        # overlaying textual information
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # fontScale
        fontScale = 0.8

        # Blue color in BGR
        color = (200, 255, 50)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.putText() method
        final = cv2.putText(final, info1 , (100,50), font,
                            fontScale, color, thickness, cv2.LINE_AA)
        final = cv2.putText(final, info2 , (100,80), font,
                            fontScale, color, thickness, cv2.LINE_AA)


        cv2.imshow("lane", final)
        # cv2.imshow("lane",dst)
        # cv2.imshow("preview", edge)
        # cv2.imshow("original", image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    frame.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()