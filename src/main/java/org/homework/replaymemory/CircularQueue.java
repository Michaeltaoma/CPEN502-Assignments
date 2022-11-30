package org.homework.replaymemory;

import java.util.LinkedList;

public class CircularQueue<T> extends LinkedList<T> {
    private int capacity = 10;

    public CircularQueue(final int capacity) {
        this.capacity = capacity;
    }

    public boolean add(final T e) {
        if (this.size() >= this.capacity) this.removeFirst();
        return super.add(e);
    }

    public Object[] toArray() {
        return super.toArray();
    }
}
