import MLX
import MLXNN

extension Module {
    /// Estimated memory footprint of all parameters in bytes.
    public func parameterMemoryBytes() -> Int {
        var total = 0
        for array in allParameters() {
            total += array.nbytes
        }
        return total
    }

    /// Replace all parameters with empty arrays to free GPU memory.
    /// After calling this, the module is unusable for inference.
    public func clearParameters() {
        apply(filter: Self.filterAll) { _ in MLXArray() }
        Memory.clearCache()
    }

    /// Collect all parameter arrays (flattened).
    private func allParameters() -> [MLXArray] {
        filterMap(filter: Self.filterAll, map: Self.mapParameters())
            .flattenedValues()
    }
}
