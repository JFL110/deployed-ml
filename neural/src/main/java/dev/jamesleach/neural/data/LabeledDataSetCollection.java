package dev.jamesleach.neural.data;

import com.google.common.collect.ImmutableList;
import lombok.Data;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Collection;
import java.util.stream.Collectors;

/**
 * Wrapper around a list of {@link LabeledDataSet}s.
 */
@Data
public class LabeledDataSetCollection {
  private final ImmutableList<LabeledDataSet> dataSets;
  private final DataShape dataShape;

  public LabeledDataSetCollection(LabeledDataSet... dataSets) {
    this(ImmutableList.copyOf(dataSets));
  }

  public LabeledDataSetCollection(Collection<LabeledDataSet> dataSets) {
    this.dataSets = ImmutableList.copyOf(dataSets);
    this.dataShape = this.dataSets.isEmpty() ? null : this.dataSets.get(0).getDataShape();
  }


  /**
   * @return a {@link DataSetIterator} that iterates over all {@link LabeledDataSet}.
   */
  public DataSetIterator toDataSetIterator() {
    return new ListDataSetIterator<>(
      dataSets.stream().map(LabeledDataSet::getDataSet).collect(Collectors.toList())
    );
  }
}
