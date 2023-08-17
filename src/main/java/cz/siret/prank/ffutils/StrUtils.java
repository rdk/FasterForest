package cz.siret.prank.ffutils;

import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

/**
 *
 */
public class StrUtils {

    private static class MSN extends ToStringStyle {
        MSN() {
            super();

            this.setContentStart("[");
            this.setFieldSeparator(System.lineSeparator() + "  ");
            this.setFieldSeparatorAtStart(true);
            this.setContentEnd(System.lineSeparator() + "]");
            this.setUseShortClassName(true);
            this.setUseIdentityHashCode(false);
        }
    }

    public static final ToStringStyle MULTILINE_SIMPLE_NAMES = new MSN();

    
    public static String toStr(Object obj) {
        return ToStringBuilder.reflectionToString(obj, MULTILINE_SIMPLE_NAMES);
    }

}
