enum Dtype {
    INT8(1), INT16(2), INT32(4), INT64(8), FLOAT32(4), FLOAT64(8);

    private final int sizeInBytes;

    Dtype(int sizeInBytes) {
        this.sizeInBytes = sizeInBytes;
    }

    public int getSizeInBytes() {
        return sizeInBytes;
    }
}