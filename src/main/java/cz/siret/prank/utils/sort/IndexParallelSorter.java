package cz.siret.prank.utils.sort;

import java.util.Arrays;
import java.util.concurrent.CountedCompleter;
import java.util.concurrent.ForkJoinPool;

/**
 *  Modified version of parallel sorting of objects from java.util.
 *  Code based on jre 1.8.0_161
 *
 *  Instead of sorting array of objects we are sorting array of integers that represents indices of obects or numbers in some other array.
 */
public class IndexParallelSorter {

    /**
     * The minimum array length below which a parallel sorting
     * algorithm will not further partition the sorting task. Using
     * smaller sizes typically results in memory contention across
     * tasks that makes parallel speedups unlikely.
     */
    private static final int MIN_ARRAY_SORT_GRAN = 1 << 13;

    /**
     * Based on Arrays.parallelSort(T[] a, Comparator<? super T> cmp)
     * implementation from 
     *
     * Sorts the specified array of objects according to the order induced by
     * the specified comparator.  All elements in the array must be
     * <i>mutually comparable</i> by the specified comparator (that is,
     * {@code c.compare(e1, e2)} must not throw a {@code ClassCastException}
     * for any elements {@code e1} and {@code e2} in the array).
     *
     * <p>This sort is guaranteed to be <i>stable</i>:  equal elements will
     * not be reordered as a result of the sort.
     *
     * @implNote The sorting algorithm is a parallel sort-merge that breaks the
     * array into sub-arrays that are themselves sorted and then merged. When
     * the sub-array length reaches a minimum granularity, the sub-array is
     * sorted using the appropriate {@link Arrays#sort(Object[]) Arrays.sort}
     * method. If the length of the specified array is less than the minimum
     * granularity, then it is sorted using the appropriate {@link
     * Arrays#sort(Object[]) Arrays.sort} method. The algorithm requires a
     * working space no greater than the size of the original array. The
     * {@link ForkJoinPool#commonPool() ForkJoin common pool} is used to
     * execute any parallel tasks.
     *
     * @param a the array to be sorted
     * @param cmp the comparator to determine the order of the array.  A
     *        {@code null} value indicates that the elements'
     *        {@linkplain Comparable natural ordering} should be used.
     * @throws ClassCastException if the array contains elements that are
     *         not <i>mutually comparable</i> using the specified comparator
     * @throws IllegalArgumentException (optional) if the comparator is
     *         found to violate the {@link java.util.Comparator} contract
     *
     * @since 1.8
     */
    public static void parallelSortIndices(int[] a, int parallelism, ForkJoinPool pool, IntComparator cmp) {
        int n = a.length;
        int p = parallelism;

        if (n <= MIN_ARRAY_SORT_GRAN || p == 1) {

            IndexTimSort.sort(a, 0, n, cmp, null, 0, 0);
        } else {

            int g = n / (p << 2);
            g = Math.max(g, MIN_ARRAY_SORT_GRAN);

            int[] w = new int[n];

            pool.invoke(new IntSort.Sorter(null, a, w,0, n, 0, g, cmp));
            //new IntSort.Sorter(null, a, w,0, n, 0, g, cmp).invoke();
        }

    }


    /**
     * A trigger for secondary merge of two merges
     */
    static final class Relay extends CountedCompleter<Void> {
        static final long serialVersionUID = 2446542900576103244L;
        final CountedCompleter<?> task;
        Relay(CountedCompleter<?> task) {
            super(null, 1);
            this.task = task;
        }
        public final void compute() { }
        public final void onCompletion(CountedCompleter<?> t) {
            task.compute();
        }
    }

    /**
     * A placeholder task for Sorters, used for the lowest
     * quartile task, that does not need to maintain array state.
     */
    static final class EmptyCompleter extends CountedCompleter<Void> {
        static final long serialVersionUID = 2446542900576103244L;
        EmptyCompleter(CountedCompleter<?> p) { super(p); }
        public final void compute() { }
    }


    /** index + Comparator support class */
    static final class IntSort {

        static final class Sorter extends CountedCompleter<Void> {

            static final long serialVersionUID = 2446542900576103244L;
            final int[] a, w;
            final int base, size, wbase, gran;
            IntComparator comparator;

            Sorter(CountedCompleter<?> par, int[] a, int[] w, int base, int size,
                   int wbase, int gran,
                   IntComparator comparator) {
                super(par);
                this.a = a; this.w = w; this.base = base; this.size = size;
                this.wbase = wbase; this.gran = gran;
                this.comparator = comparator;
            }

            public final void compute() {
                CountedCompleter<?> s = this;
                IntComparator c = this.comparator;
                int[] a = this.a, w = this.w; // localize all params
                int b = this.base, n = this.size, wb = this.wbase, g = this.gran;
                while (n > g) {
                    int h = n >>> 1, q = h >>> 1, u = h + q; // quartiles
                    Relay fc = new Relay(new IntSort.Merger(s, w, a, wb, h,
                        wb+h, n-h, b, g, c));
                    Relay rc = new Relay(new IntSort.Merger(fc, a, w, b+h, q,
                        b+u, n-u, wb+h, g, c));
                    new IntSort.Sorter(rc, a, w, b+u, n-u, wb+u, g, c).fork();
                    new IntSort.Sorter(rc, a, w, b+h, q, wb+h, g, c).fork();;
                    Relay bc = new Relay(new IntSort.Merger(fc, a, w, b, q,
                        b+q, h-q, wb, g, c));
                    new IntSort.Sorter(bc, a, w, b+q, h-q, wb+q, g, c).fork();
                    s = new EmptyCompleter(bc);
                    n = q;
                }
                IndexTimSort.sort(a, b, b + n, c, w, wb, n);
                s.tryComplete();
            }
        }

        static final class Merger extends CountedCompleter<Void> {
            static final long serialVersionUID = 2446542900576103244L;
            final int[] a, w; // main and workspace arrays
            final int lbase, lsize, rbase, rsize, wbase, gran;
            IntComparator comparator;

            Merger(CountedCompleter<?> par, int[] a, int[] w,
                   int lbase, int lsize, int rbase,
                   int rsize, int wbase, int gran,
                   IntComparator comparator) {
                super(par);
                this.a = a; this.w = w;
                this.lbase = lbase; this.lsize = lsize;
                this.rbase = rbase; this.rsize = rsize;
                this.wbase = wbase; this.gran = gran;
                this.comparator = comparator;
            }

            public final void compute() {
                IntComparator c = this.comparator;
                int[] a = this.a, w = this.w; // localize all params
                int lb = this.lbase, ln = this.lsize, rb = this.rbase,
                    rn = this.rsize, k = this.wbase, g = this.gran;
                if (a == null || w == null || lb < 0 || rb < 0 || k < 0 ||
                    c == null)
                    throw new IllegalStateException(); // hoist checks
                for (int lh, rh;;) {  // split larger, find point in smaller
                    if (ln >= rn) {
                        if (ln <= g)
                            break;
                        rh = rn;
                        int split = a[(lh = ln >>> 1) + lb];
                        for (int lo = 0; lo < rh; ) {
                            int rm = (lo + rh) >>> 1;
                            if (c.compare(split, a[rm + rb]) <= 0)
                                rh = rm;
                            else
                                lo = rm + 1;
                        }
                    }
                    else {
                        if (rn <= g)
                            break;
                        lh = ln;
                        int split = a[(rh = rn >>> 1) + rb];
                        for (int lo = 0; lo < lh; ) {
                            int lm = (lo + lh) >>> 1;
                            if (c.compare(split, a[lm + lb]) <= 0)
                                lh = lm;
                            else
                                lo = lm + 1;
                        }
                    }
                    IntSort.Merger m = new IntSort.Merger(this, a, w, lb + lh, ln - lh,
                        rb + rh, rn - rh,
                        k + lh + rh, g, c);
                    rn = rh;
                    ln = lh;
                    addToPendingCount(1);
                    m.fork();
                }

                int lf = lb + ln, rf = rb + rn; // index bounds
                while (lb < lf && rb < rf) {
                    int t, al, ar;
                    if (c.compare((al = a[lb]), (ar = a[rb])) <= 0) {
                        lb++; t = al;
                    }
                    else {
                        rb++; t = ar;
                    }
                    w[k++] = t;
                }
                if (rb < rf)
                    System.arraycopy(a, rb, w, k, rf - rb);
                else if (lb < lf)
                    System.arraycopy(a, lb, w, k, lf - lb);

                tryComplete();
            }

        }
    } // FJObject

}
