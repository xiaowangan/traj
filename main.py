"""Create and run the migrated desktop application; all imports are local to traj."""
def main():
    from polishing.ui.bootstrap import main as run_application
    return run_application()

if __name__ == "__main__":
    main()
