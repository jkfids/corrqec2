import sinter


def save_stats_to_csv(stats: sinter.TaskStats, filepath: str):
    """Save collected Sinter TaskStats to a CSV file."""
    with open(filepath, "w") as f:
        print(sinter.CSV_HEADER, file=f)
        for stat in stats:
            print(stat.to_csv_line(), file=f)
