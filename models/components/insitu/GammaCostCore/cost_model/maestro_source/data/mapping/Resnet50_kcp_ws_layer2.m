
Network Resnet50 {
        Layer CONV1 {
            Type: CONV
                Stride { X: 1, Y: 1 }
            Dimensions { K: 1, C: 1, R: 3, S: 3, Y:224, X:224 }
            Dataflow {
                TempInterMap(Sz(R),1) Y;
                TempInterMap(Sz(S),1) X;
                TemporalMap(1,1) C;
                SpatialMap(1,1) K;
                TemporalMap(Sz(R),Sz(R)) R;
                TemporalMap(Sz(S),Sz(S)) S;
                Cluster(1, P);
                SpatialMap(1,1) C;
                TemporalMap(Sz(R),1) Y;
                TemporalMap(Sz(S),1) X;
                TemporalMap(Sz(R),Sz(R)) R;
                TemporalMap(Sz(S),Sz(S)) S;
            }
        }
}