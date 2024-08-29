import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import jcuda.*;
import jcuda.jcublas.*;
import jcuda.runtime.*;

public class Tensor {
    private int rows;
    private int columns;
    private Dtype dtype;
    private ByteBuffer data;
    private Pointer devicePointer;
    private boolean isOnGPU;

    public Tensor(int rows, int columns, Dtype dtype) {
        this.rows = rows;
        this.columns = columns;
        this.dtype = dtype;
        this.data = ByteBuffer.allocateDirect(rows * columns * dtype.getSizeInBytes()).order(ByteOrder.nativeOrder());
        this.isOnGPU = false;
    }

    public void set(int row, int col, Number value) {
        int index = (row * columns + col) * dtype.getSizeInBytes();
        switch (dtype) {
            case INT8 -> data.put(index, value.byteValue());
            case INT16 -> data.putShort(index, value.shortValue());
            case INT32 -> data.putInt(index, value.intValue());
            case INT64 -> data.putLong(index, value.longValue());
            case FLOAT32 -> data.putFloat(index, value.floatValue());
            case FLOAT64 -> data.putDouble(index, value.doubleValue());
        }
    }

    public Number get(int row, int col) {
        int index = (row * columns + col) * dtype.getSizeInBytes();
        return switch (dtype) {
            case INT8 -> data.get(index);
            case INT16 -> data.getShort(index);
            case INT32 -> data.getInt(index);
            case INT64 -> data.getLong(index);
            case FLOAT32 -> data.getFloat(index);
            case FLOAT64 -> data.getDouble(index);
        };
    }

    public void changeDtype(Dtype newDtype) {
        if (newDtype == this.dtype) {
            return;
        }

        ByteBuffer newData = ByteBuffer.allocateDirect(rows * columns * newDtype.getSizeInBytes()).order(ByteOrder.nativeOrder());

        for (int i = 0; i < rows * columns; i++) {
            Number value = get(i / columns, i % columns);
            switch (newDtype) {
                case INT8 -> newData.put(i * newDtype.getSizeInBytes(), value.byteValue());
                case INT16 -> newData.putShort(i * newDtype.getSizeInBytes(), value.shortValue());
                case INT32 -> newData.putInt(i * newDtype.getSizeInBytes(), value.intValue());
                case INT64 -> newData.putLong(i * newDtype.getSizeInBytes(), value.longValue());
                case FLOAT32 -> newData.putFloat(i * newDtype.getSizeInBytes(), value.floatValue());
                case FLOAT64 -> newData.putDouble(i * newDtype.getSizeInBytes(), value.doubleValue());
            }
        }

        this.data = newData;
        this.dtype = newDtype;

        if (isOnGPU) {
            unloadFromGPU();
            loadToGPU();
        }
    }

    public void loadToGPU() {
        if (!isOnGPU) {
            JCuda.initialize();
            devicePointer = new Pointer();
            JCuda.cudaMalloc(devicePointer, rows * columns * dtype.getSizeInBytes());
            JCuda.cudaMemcpy(devicePointer, Pointer.to(data), rows * columns * dtype.getSizeInBytes(), cudaMemcpyKind.cudaMemcpyHostToDevice);
            isOnGPU = true;
        }
    }

    public void unloadFromGPU() {
        if (isOnGPU) {
            JCuda.cudaMemcpy(Pointer.to(data), devicePointer, rows * columns * dtype.getSizeInBytes(), cudaMemcpyKind.cudaMemcpyDeviceToHost);
            JCuda.cudaFree(devicePointer);
            isOnGPU = false;
        }
    }

    public Tensor add(Tensor other) {
        if (rows != other.rows || columns != other.columns) {
            throw new IllegalArgumentException("Tensor dimensions must match for addition");
        }

        Tensor result = new Tensor(rows, columns, dtype);

        if (isOnGPU && other.isOnGPU) {
            result.loadToGPU();
            cublasHandle handle = new cublasHandle();
            JCublas2.cublasCreate(handle);

            float alpha = 1.0f;
            int elements = rows * columns;

            switch (dtype) {
                case FLOAT32 -> JCublas2.cublasSaxpy(handle, elements, Pointer.to(new float[]{alpha}), devicePointer, 1, other.devicePointer, 1);
                case FLOAT64 -> JCublas2.cublasDaxpy(handle, elements, Pointer.to(new double[]{alpha}), devicePointer, 1, other.devicePointer, 1);
                default -> throw new UnsupportedOperationException("GPU addition only supported for FLOAT32 and FLOAT64");
            }

            JCublas2.cublasDestroy(handle);
            result.unloadFromGPU();
        } else {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    Number sum = ((Number) get(i, j)).doubleValue() + ((Number) other.get(i, j)).doubleValue();
                    result.set(i, j, sum);
                }
            }
        }

        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            sb.append("[");
            for (int j = 0; j < columns; j++) {
                sb.append(get(i, j));
                if (j < columns - 1) sb.append(", ");
            }
            sb.append("]\n");
        }
        return sb.toString();
    }

    public static Tensor randn(int rows, int columns, Dtype dtype) {
        Tensor tensor = new Tensor(rows, columns, dtype);
        Random rand = new Random();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                double value = rand.nextGaussian();
                tensor.set(i, j, value);
            }
        }

        return tensor;
    }

    public static Tensor arange(Number start, Number end, Number step, Dtype dtype) {
        int size = (int) Math.ceil((end.doubleValue() - start.doubleValue()) / step.doubleValue());
        Tensor tensor = new Tensor(size, 1, dtype);

        for (int i = 0; i < size; i++) {
            double value = start.doubleValue() + i * step.doubleValue();
            tensor.set(i, 0, value);
        }

        return tensor;
    }

    public static Tensor linspace(Number start, Number end, int num, Dtype dtype) {
        Tensor tensor = new Tensor(num, 1, dtype);
        double step = (end.doubleValue() - start.doubleValue()) / (num - 1);

        for (int i = 0; i < num; i++) {
            double value = start.doubleValue() + i * step;
            tensor.set(i, 0, value);
        }

        return tensor;
    }

    public static void main(String[] args) {
        Tensor randTensor = Tensor.randn(2, 3, Dtype.FLOAT32);
        System.out.println("Random Tensor:");
        System.out.println(randTensor);

        Tensor arangeTensor = Tensor.arange(0, 10, 2, Dtype.INT32);
        System.out.println("Arange Tensor:");
        System.out.println(arangeTensor);

        Tensor linspaceTensor = Tensor.linspace(0, 1, 5, Dtype.FLOAT32);
        System.out.println("Linspace Tensor:");
        System.out.println(linspaceTensor);

        // GPU example
        Tensor a = Tensor.randn(3, 3, Dtype.FLOAT32);
//        Tensor b = Tensor.randn(1000, 1000, Dtype.FLOAT32);

        System.out.println(a);
//        a.loadToGPU();
//        b.loadToGPU();
//        long start = System.currentTimeMillis();
//        Tensor c = a.add(b);
//        long end = System.currentTimeMillis();
//        System.out.println("GPU addition time: " + (end - start) + "ms");
    }

}