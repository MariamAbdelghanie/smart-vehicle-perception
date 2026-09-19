import numpy as np

class ByteTrackerWrapper:
    def __init__(self, iou_threshold=0.3, max_lost=30):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks = {}
        self._next_id = 1

    def update(self, detections, frame_shape):
        if not detections:
            to_del = [t for t in self.tracks if self.tracks[t]["lost"] > self.max_lost]
            for t in to_del: del self.tracks[t]
            for t in self.tracks: self.tracks[t]["lost"] += 1
            return []
        det_boxes = np.array([[d[0],d[1],d[2],d[3]] for d in detections])
        if not self.tracks:
            results = []
            for box in det_boxes:
                tid = self._next_id; self._next_id += 1
                self.tracks[tid] = {"box": box, "lost": 0}
                results.append((*box.tolist(), tid))
            return results
        track_ids = list(self.tracks.keys())
        track_boxes = np.array([self.tracks[t]["box"] for t in track_ids])
        iou_mat = self._iou_matrix(track_boxes, det_boxes)
        matched_t = set(); matched_d = set(); results = []
        for _ in range(min(len(track_ids), len(det_boxes))):
            if iou_mat.size == 0: break
            ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            if iou_mat[ti, di] < self.iou_threshold: break
            tid = track_ids[ti]
            self.tracks[tid] = {"box": det_boxes[di], "lost": 0}
            matched_t.add(ti); matched_d.add(di)
            results.append((*det_boxes[di].tolist(), tid))
            iou_mat[ti,:] = -1; iou_mat[:,di] = -1
        for di in range(len(det_boxes)):
            if di not in matched_d:
                tid = self._next_id; self._next_id += 1
                self.tracks[tid] = {"box": det_boxes[di], "lost": 0}
                results.append((*det_boxes[di].tolist(), tid))
        to_del = []
        for ti, tid in enumerate(track_ids):
            if ti not in matched_t:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost: to_del.append(tid)
        for tid in to_del: del self.tracks[tid]
        return results

    @staticmethod
    def _iou_matrix(boxes_a, boxes_b):
        iou = np.zeros((len(boxes_a), len(boxes_b)))
        for i, a in enumerate(boxes_a):
            for j, b in enumerate(boxes_b):
                xi1=max(a[0],b[0]); yi1=max(a[1],b[1])
                xi2=min(a[2],b[2]); yi2=min(a[3],b[3])
                inter=max(0,xi2-xi1)*max(0,yi2-yi1)
                union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-6
                iou[i,j]=inter/union
        return iou