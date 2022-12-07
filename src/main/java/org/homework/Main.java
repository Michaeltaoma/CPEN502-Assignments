package org.homework;

import org.homework.robot.model.Action;

public class Main {
    public static void main(final String[] args) {
        final StringBuilder sb = new StringBuilder();
        for (final Action action : Action.values()) {
            sb.append(String.format("%s, ", action.name()));
        }
        System.out.printf(sb.toString());
    }
}
