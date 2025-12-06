package com.neocoretechs.roscar;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Arrays;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

final class ArrayFloatTensor extends FloatTensor implements Externalizable, Comparable {
	public static boolean DEBUG = false;
    float[] values;
    
    public ArrayFloatTensor() {}
    
    ArrayFloatTensor(float[] values) {
        this.values = values;
    }

    public static FloatTensor allocate(int... dims) {
        int numberOfElements = FloatTensor.numberOfElements(dims);
        return new ArrayFloatTensor(new float[numberOfElements]);
    }

    @Override
    public int size() {
        return values.length;
    }

    @Override
    public float getFloat(int index) {
        return values[index];
    }

    @Override
    public void setFloat(int index, float value) {
        values[index] = value;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

    @Override
    public FloatTensor fillInPlace(int thisOffset, int size, float value) {
        Arrays.fill(values, thisOffset, thisOffset + size, value);
        return this;
    }

    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromArray(species, values, index);
    }
    
	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(values.length);
		for(float v: values)
			out.writeFloat(v);	
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		int vsize = in.readInt();
		values = new float[vsize];
		for(int i = 0; i < vsize; i++)
			values[i]= in.readFloat();
	}

	@Override
	public int compareTo(Object o) {
		return Arrays.compare(values,((ArrayFloatTensor)o).values);
	}

}


