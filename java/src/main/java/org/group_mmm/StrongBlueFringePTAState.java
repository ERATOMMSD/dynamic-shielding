package org.group_mmm;

import de.learnlib.datastructure.pta.pta.AbstractBlueFringePTAState;
import de.learnlib.datastructure.pta.pta.BlueFringePTAState;
import de.learnlib.datastructure.pta.pta.RedBlueMerge;
import lombok.Getter;
import net.automatalib.commons.smartcollections.ArrayStorage;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

public class StrongBlueFringePTAState<SP, TP> extends AbstractBlueFringePTAState<SP, TP, StrongBlueFringePTAState<SP, TP>> {
    Set<StrongBlueFringePTAState<SP, TP>> incompatibleStates = new HashSet<>();
    @Getter
    protected int height;

    public StrongBlueFringePTAState() {
        this.height = 0;
        this.promoteHeight();
    }

    void setParent(StrongBlueFringePTAState<SP, TP> parent) {
        this.parent = parent;
    }

    StrongBlueFringePTAState<SP, TP> getParent() {
        return this.parent;
    }

    int getParentInput() {
        return this.parentInput;
    }

    SP getProperty() {
        return this.property;
    }

    ArrayStorage<TP> getTransPropertiesRefl() {
        return this.transProperties;
    }

    boolean getIsCopy() {
        return this.isCopy;
    }

    int getId() {
        return this.id;
    }

    ArrayStorage<StrongBlueFringePTAState<SP, TP>> getSuccessors() {
        return this.successors;
    }

    void setProperty(SP property) {
        this.property = property;
    }

    void setHeight(int newHeight) {
        if (newHeight > this.height) {
            this.height = newHeight;
            this.promoteHeight();
        }
    }

    protected void promoteHeight() {
        if (Objects.nonNull(this.parent)) {
            this.parent.setHeight(this.height + 1);
        }
    }

    void setIncompatibleState(StrongBlueFringePTAState<SP, TP> incompatibleState) {
        if (!incompatibleStates.contains(incompatibleState)) {
            incompatibleStates.add(incompatibleState);
            promoteIncompatible(incompatibleState);
        }
    }

    boolean isIncompatible(StrongBlueFringePTAState<SP, TP> state) {
        return this.incompatibleStates.contains(state);
    }

    protected void promoteIncompatible(StrongBlueFringePTAState<SP, TP> incompatibleState) {
        if (Objects.nonNull(this.parent) && Objects.nonNull(incompatibleState.parent)) {
            this.parent.setIncompatibleState(incompatibleState.parent);
        }
    }

    @Override
    protected StrongBlueFringePTAState<SP, TP> createState() {
        return new StrongBlueFringePTAState<>();
    }
}
