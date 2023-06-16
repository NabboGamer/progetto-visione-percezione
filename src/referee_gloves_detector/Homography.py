import numpy as np
import cv2
import pandas
import math
import os


class Homography():

    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

        self.src_copy = self.src.copy()
        self.dst_copy = self.dst.copy()
        self.src_x, self.src_y = -1, -1
        self.dst_x, self.dst_y = -1, -1

        self.src_list = []
        self.dst_list = []

        self.drawing = False


        self.H = []

    def normalize_points(self,points_virtual_pitch, points_real_pitch):

        def get_normalization_matrix(pts, name="A"):
            pts = np.array(pts).astype(np.float64)
            x_mean, y_mean = np.mean(pts, axis=0)
            var_x, var_y = np.var(pts, axis=0)

            s_x, s_y = np.sqrt(2 / var_x), np.sqrt(2 / var_y)

            print("Matrix: {4} : meanx {0}, meany {1}, varx {2}, vary {3}, sx {5}, sy {6} ".format(x_mean, y_mean, var_x,
                                                                                                   var_y, name, s_x, s_y))

            n = np.array([[s_x, 0, -s_x * x_mean], [0, s_y, -s_y * y_mean], [0, 0, 1]])
            # print(n)

            n_inv = np.array([[1. / s_x, 0, x_mean], [0, 1. / s_y, y_mean], [0, 0, 1]])
            return n.astype(np.float64), n_inv.astype(np.float64)

        ret_correspondences = []
        imp, objp = points_virtual_pitch, points_real_pitch
        N_x, N_x_inv = get_normalization_matrix(objp, "A")
        N_u, N_u_inv = get_normalization_matrix(imp, "B")

        hom_imp = np.array([[[each[0]], [each[1]], [1.0]] for each in imp])
        hom_objp = np.array([[[each[0]], [each[1]], [1.0]] for each in objp])

        normalized_hom_imp = hom_imp
        normalized_hom_objp = hom_objp

        for i in range(normalized_hom_objp.shape[0]):
            n_o = np.matmul(N_x, normalized_hom_objp[i])
            normalized_hom_objp[i] = n_o / n_o[-1]

            n_u = np.matmul(N_u, normalized_hom_imp[i])
            normalized_hom_imp[i] = n_u / n_u[-1]

        normalized_objp = normalized_hom_objp.reshape(normalized_hom_objp.shape[0], normalized_hom_objp.shape[1])
        normalized_imp = normalized_hom_imp.reshape(normalized_hom_imp.shape[0], normalized_hom_imp.shape[1])

        normalized_objp = normalized_objp[:, :-1]
        normalized_imp = normalized_imp[:, :-1]


        ret_correspondences = (imp, objp, normalized_imp, normalized_objp, N_u, N_x, N_u_inv, N_x_inv)

        return ret_correspondences

    def _from_detection(self):
        """
            Semi-automatic detection of the pairs of points to be used to 
            compute the homography matrix. Homography source points are 
            automatically detected, destination points has to be manually 
            selected by the user.
        """
        cv2.namedWindow("Homography - SRC")
        cv2.moveWindow("Homography - SRC",  80, 80)

        cv2.namedWindow("Homography - DST")
        cv2.moveWindow("Homography - DST", 780, 80)
        cv2.setMouseCallback("Homography - DST", self._select_points_dst)

        background = self.src
        dh, dw, _ = background.shape

        keypoints = pandas.read_csv("imgs/keypoints.csv")
        keypoints = keypoints.dropna()
        
        for _, row in keypoints.iterrows():
            x, y = row['x'], row['y']
            print(x)
            print(y)
            self.src_list.append([int(x), int(y)])
            cv2.circle(self.src_copy, (int(x), int(y)), 0, (0, 0, 255), 10)

            while True:
                cv2.imshow("Homography - SRC", self.src_copy)
                cv2.imshow("Homography - DST", self.dst_copy)

                k = cv2.waitKey(1) & 0xFF
                if k == ord("s"):
                    cv2.circle(self.src_copy, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.circle(self.dst_copy, (self.dst_x, self.dst_y), 5, (0, 255, 0), -1)

                    self.dst_list.append([self.dst_x, self.dst_y])

                    break
                elif k == ord("q"):
                    os._exit(0)
        cv2.destroyAllWindows()

        corr = self.normalize_points(self.dst_list, self.src_list)
        self._compute_view_based_homography(corr)

    def _compute_view_based_homography(self, correspondence, reproj=True):
        image_points = correspondence[0]
        object_points = correspondence[1]
        normalized_image_points = correspondence[2]
        normalized_object_points = correspondence[3]
        N_u = correspondence[4]
        N_x = correspondence[5]
        N_u_inv = correspondence[6]
        N_x_inv = correspondence[7]

        N = len(image_points)
        print("Number of points in current view : ", N)

        M = np.zeros((2 * N, 9), dtype=np.float64)
        print("Shape of Matrix M : ", M.shape)

        print("N_model\n", N_x)
        print("N_observed\n", N_u)

        # create row wise allotment for each 0-2i rows
        # that means 2 rows..
        for i in range(N):
            X, Y = normalized_object_points[i]  # A
            u, v = normalized_image_points[i]  # B

            row_1 = np.array([-X, -Y, -1, 0, 0, 0, X * u, Y * u, u])
            row_2 = np.array([0, 0, 0, -X, -Y, -1, X * v, Y * v, v])
            M[2 * i] = row_1
            M[(2 * i) + 1] = row_2

            print("p_model {0} \t p_obs {1}".format((X, Y), (u, v)))

        u, s, vh = np.linalg.svd(M)
        print("Computing SVD of M")


        h_norm = vh[np.argmin(s)]
        h_norm = h_norm.reshape(3, 3)
        print(N_u_inv)
        print(N_x)
        # h = h_norm
        h = np.matmul(np.matmul(N_u_inv, h_norm), N_x)

        # if abs(h[2, 2]) > 10e-8:
        h = h[:, :] / h[2, 2]

        print("Homography for View : \n", h)

        if reproj:
            reproj_error = 0
            for i in range(len(image_points)):
                t1 = np.array([[object_points[i][0]], [object_points[i][1]], [1.0]])
                t = np.matmul(h, t1).reshape(1, 3)
                t = t / t[0][-1]
                formatstring = "Imp {0} | ObjP {1} | Tx {2}".format(image_points[i], object_points[i], t)
                print(formatstring)
                reproj_error += np.sum(np.abs(image_points[i] - t[0][:-1]))
            reproj_error = np.sqrt(reproj_error / N) / 100.0
            print("Reprojection error : ", reproj_error)

        self.H = h

    def _select_points_src(self, event, x, y, flags, params):
        """
            Callback function called when the user select a point
            on the homography source window.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.src_x, self.src_y = x, y
            cv2.circle(self.src_copy, (x, y), 5, (0, 0, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def _select_points_dst(self, event, x, y, flags, params):
        """
            Callback function called when the user select a point
            on the homography destination window.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.dst_x, self.dst_y = x, y
            cv2.circle(self.dst_copy, (x, y), 5, (0, 0, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False   